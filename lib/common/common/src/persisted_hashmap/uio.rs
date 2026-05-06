use std::io::{self, Cursor};
use std::marker::PhantomData;
use std::mem::size_of;
use std::path::Path;

use aligned_vec::AVec;
use ph::fmph::Function;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use super::{BucketOffset, Header, Key, PartialEntry, PartialEntryKind, ValuesLen};
use crate::generic_consts::{Random, Sequential};
use crate::iterator_ext::ordering_iterator::OrderingIterator;
use crate::universal_io::{
    OpenOptions, ReadRange, Result, UniversalIoError, UniversalRead, UniversalReadPipeline,
};

/// On-disk hash map accessed via [`UniversalRead`].
pub struct UniversalHashMap<K, V, S>
where
    K: Key + ?Sized,
    V: Sized + Copy + FromBytes + Immutable + IntoBytes + KnownLayout,
    S: UniversalRead<u8>,
{
    storage: S,
    header: Header,
    phf: Function,
    /// Absolute byte offset where entry data begins (right after the bucket offsets array).
    entries_start: u64,
    phantom: PhantomData<(V, K)>,
}

impl<'key, K, V, S> UniversalHashMap<K, V, S>
where
    K: Key + ?Sized + 'key,
    V: Sized + Copy + FromBytes + Immutable + IntoBytes + KnownLayout,
    S: UniversalRead<u8>,
{
    /// Load the hash map from file.
    pub fn open(path: impl AsRef<Path>, options: OpenOptions) -> Result<Self> {
        let storage = S::open(path, options)?;

        // 1. Read header.
        let header_bytes = storage.read::<Sequential>(ReadRange {
            byte_offset: 0,
            length: size_of::<Header>() as u64,
        })?;
        let (header, _) = Header::read_from_prefix(&header_bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid header"))?;

        if header.key_type != K::NAME {
            return Err(UniversalIoError::from(io::Error::new(
                io::ErrorKind::InvalidData,
                "Key type mismatch",
            )));
        }

        // 2. Read PHF. The region between the header and buckets_pos contains the
        //    serialised PHF followed by padding; `Function::read` consumes only what
        //    it needs and ignores trailing bytes.
        let phf_region_start = size_of::<Header>() as u64;
        let phf_region_len = header
            .buckets_pos
            .checked_sub(phf_region_start)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "buckets_pos before header end")
            })?;
        let phf_bytes = storage.read::<Sequential>(ReadRange {
            byte_offset: phf_region_start,
            length: phf_region_len,
        })?;
        let phf = Function::read(&mut Cursor::new(&*phf_bytes))?;

        let entries_start =
            header.buckets_pos + header.buckets_count * size_of::<BucketOffset>() as u64;

        Ok(UniversalHashMap {
            storage,
            header,
            phf,
            entries_start,
            phantom: PhantomData,
        })
    }

    /// Number of distinct keys stored in the hash map.
    pub fn keys_count(&self) -> usize {
        self.header.buckets_count as usize
    }

    pub fn for_each_key<E: From<UniversalIoError>>(
        &self,
        mut f: impl FnMut(&K) -> Result<(), E>,
    ) -> Result<(), E> {
        let bucket_count = self.header.buckets_count as usize;
        let bytes = self.storage.read::<Sequential>(ReadRange {
            byte_offset: self.header.buckets_pos,
            length: (bucket_count * size_of::<BucketOffset>()) as u64,
        })?;
        let mut offsets: Vec<BucketOffset> = bytes
            .chunks_exact(size_of::<BucketOffset>())
            .map(|c| BucketOffset::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        offsets.sort_unstable();
        self.for_each_sparse(
            PartialEntryKind::KeyOnly,
            offsets.into_iter().map(|o| ((), Request::Offset(o))),
            |(), entry| {
                // TODO: use method
                let key = match entry {
                    Some(
                        PartialEntry::Key(k)
                        | PartialEntry::KeyAndValuesLen(k, _)
                        | PartialEntry::KeyAndValues(k, _, _),
                    ) => k,
                    _ => unreachable!(),
                };
                f(key)
            },
        )
    }

    pub fn batch_with_entry<Meta, E: From<UniversalIoError>>(
        &self,
        keys: impl IntoIterator<Item = (Meta, &'key K)>,
        mut f: impl FnMut(Meta, Option<&[V]>) -> Result<(), E>,
    ) -> Result<(), E> {
        self.for_each_sparse(
            PartialEntryKind::KeyAndValues,
            keys.into_iter()
                .map(|(meta, key)| (meta, Request::Key(key))),
            |meta, entry| {
                let values = entry.map(|e| match e {
                    PartialEntry::KeyAndValues(_, values, _) => values,
                    _ => unreachable!(),
                });
                f(meta, values)
            },
        )
    }

    fn for_each_sparse<Meta, E: From<UniversalIoError>>(
        &self,
        entry_kind: PartialEntryKind,
        requests: impl Iterator<Item = (Meta, Request<'key, K>)>,
        mut f: impl FnMut(Meta, Option<PartialEntry<'_, K, V>>) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut sparse = SparsePipeline::new(self, entry_kind)?;
        let mut pipeline = S::ReadPipeline::<'_, Entry<'key, Meta, K>>::new()?;
        let mut requests = requests.into_iter();
        loop {
            while pipeline.can_schedule() {
                let Some((entry, range)) = sparse.refill(&mut requests, &mut f)? else {
                    break;
                };
                pipeline.schedule::<Random>(entry, &self.storage, range)?;
            }
            let Some((entry, data)) = pipeline.wait()? else {
                break;
            };
            sparse.process(entry, &data, &mut f)?;
        }
        Ok(())
    }

    pub fn for_each_entry<E: From<UniversalIoError>>(
        &self,
        mut f: impl FnMut(&K, &[V]) -> Result<(), E>,
    ) -> Result<(), E> {
        let file_len = self.storage.len()?;

        let range = ReadRange {
            byte_offset: self.entries_start,
            length: file_len - self.entries_start,
        };

        let align = K::ALIGN.max(size_of::<ValuesLen>()).max(size_of::<V>());
        let mut buf = AVec::<u8>::new(align);
        // `parse` skips inter-entry padding via the memory pointer's alignment, so
        // `view.as_ptr() % K::ALIGN` must track the live data's file-offset modulo. To preserve
        // that across compaction, only drop K::ALIGN-aligned chunks from the front of `buf`;
        // any sub-K::ALIGN remainder stays as a prefix and `view_start` marks where live
        // data begins.
        let mut view_start = 0usize;

        let iter = OrderingIterator::new(
            self.storage
                .read_iter::<Sequential, usize>(range.iter_autochunks::<u8>().enumerate())?,
        );

        for record in iter {
            let (_, mini_buf) = record?;
            buf.extend_from_slice(&mini_buf);

            let mut view: &[u8] = &buf[view_start..];
            loop {
                match PartialEntry::parse(view).unwrap(/* TODO */) {
                    PartialEntry::KeyAndValues(key, values, leftover) => {
                        f(key, values)?;
                        view = leftover;
                    }
                    _ => break,
                }
            }

            let remaining = view.len();
            view_start = buf.len() - remaining;

            // Compact: drop a K::ALIGN-aligned prefix from `buf`, keeping the rest (and the
            // alignment invariant) intact.
            let drop_len = view_start - view_start % K::ALIGN;
            if drop_len > 0 {
                buf.copy_within(drop_len.., 0);
                buf.truncate(buf.len() - drop_len);
                view_start -= drop_len;
            }
        }

        if buf.len() > view_start {
            return Err(uio_data_err("Trailing bytes left after parsing all entries").into());
        }

        Ok(())
    }

    pub fn get(&self, key: &K) -> Result<Option<Vec<V>>> {
        let mut result: Option<Vec<V>> = None;
        self.for_each_sparse(
            PartialEntryKind::KeyAndValues,
            std::iter::once(((), Request::Key(key))),
            |(), entry| -> Result<()> {
                result = entry.map(|e| e.values().expect("TODO").to_vec());
                Ok(())
            },
        )?;
        Ok(result)
    }

    /// Return the number of values for `key` *without* reading the values themselves.
    pub fn get_values_count(&self, key: &K) -> Result<Option<usize>> {
        let mut result: Option<usize> = None;
        self.for_each_sparse(
            PartialEntryKind::KeyAndValuesLen,
            std::iter::once(((), Request::Key(key))),
            |(), entry| -> Result<()> {
                result = entry.map(|e| e.values_len().expect("TODO") as usize);
                Ok(())
            },
        )?;
        Ok(result)
    }

    /// Populate the RAM cache for the backing file.
    pub fn populate(&self) -> Result<()> {
        self.storage.populate()
    }

    /// Evict the backing file data from RAM cache.
    pub fn clear_ram_cache(&self) -> Result<()> {
        self.storage.clear_ram_cache()
    }
}

fn uio_data_err(msg: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> UniversalIoError {
    UniversalIoError::Io(io::Error::new(io::ErrorKind::InvalidData, msg))
}

fn parse_entry_offset(data: &[u8]) -> Result<BucketOffset> {
    Ok(BucketOffset::from_ne_bytes(
        data.try_into()
            .map_err(|_| uio_data_err("Can't read bucket offset"))?,
    ))
}

/// State machine driver for [`UniversalHashMap::for_each_sparse_impl2`]. Owns
/// the queue of entries waiting for an I/O slot and exposes:
/// [`Self::refill`] — produce the next entry to schedule —
/// and [`Self::process`] — consume a completed read.
/// The caller drives the underlying I/O pipeline.
struct SparsePipeline<'map, 'key, Meta, K, V, R>
where
    K: Key + ?Sized + 'key,
    V: Sized + Copy + FromBytes + Immutable + IntoBytes + KnownLayout,
    R: UniversalRead<u8>,
{
    map: &'map UniversalHashMap<K, V, R>,
    /// Specifies which entry fields we are interested in. E.g., for
    /// [`PartialEntryKind::KeyOnly`], read just enough to parse the key
    /// and skip the values.
    entry_kind: PartialEntryKind,
    to_schedule: Vec<Entry<'key, Meta, K>>,
    entry_read_size_est: u64,
    file_len: u64,
}

impl<'map, 'key, Meta, K, V, R> SparsePipeline<'map, 'key, Meta, K, V, R>
where
    K: Key + ?Sized + 'key,
    V: Copy + FromBytes + Immutable + IntoBytes + KnownLayout,
    R: UniversalRead<u8>,
{
    fn new(map: &'map UniversalHashMap<K, V, R>, entry_kind: PartialEntryKind) -> Result<Self> {
        let entry_read_size_est = entry_kind.estimated_size::<K, V>() as u64;
        let file_len = map.storage.len()?;
        Ok(Self {
            map,
            entry_kind,
            to_schedule: Vec::new(),
            entry_read_size_est,
            file_len,
        })
    }

    /// Produce the next entry to schedule.
    fn refill<E, F>(
        &mut self,
        requests: &mut impl Iterator<Item = (Meta, Request<'key, K>)>,
        f: &mut F,
    ) -> Result<Option<(Entry<'key, Meta, K>, ReadRange)>, E>
    where
        E: From<UniversalIoError>,
        F: FnMut(Meta, Option<PartialEntry<'_, K, V>>) -> Result<(), E>,
    {
        if let Some(entry) = self.to_schedule.pop() {
            match &entry.state {
                EntryState::ReadingOffset { .. } => {
                    unreachable!("only Loading entries are queued in to_schedule")
                }

                EntryState::ReadingEntry {
                    byte_offset,
                    buf,
                    expected_len,
                    requested_key: _,
                } => {
                    let range = ReadRange::new(
                        byte_offset.saturating_add(buf.len() as u64),
                        expected_len.saturating_sub(buf.len() as u64),
                    );
                    return Ok(Some((entry, range.clamp::<u8>(self.file_len))));
                }
            };
        }

        while let Some((meta, request)) = requests.next() {
            let (state, range);
            match request {
                Request::Offset(offset) => {
                    // Offset request: location is known, jump straight to Loading.
                    let byte_offset = self.map.entries_start + offset;
                    state = EntryState::ReadingEntry {
                        byte_offset,
                        buf: Vec::new(),
                        expected_len: self.entry_read_size_est,
                        requested_key: None,
                    };
                    range = ReadRange {
                        byte_offset,
                        length: self.entry_read_size_est,
                    };
                }
                Request::Key(requested_key) => {
                    // PHF miss: no stored entry; report immediately and continue.
                    let Some(hash) = self.map.phf.get(requested_key) else {
                        f(meta, None)?;
                        continue;
                    };
                    // PHF hit: schedule the bucket-offset read; transitions to
                    // Loading once the offset arrives.
                    let bucket_byte_offset =
                        self.map.header.buckets_pos + hash * size_of::<BucketOffset>() as u64;
                    state = EntryState::ReadingOffset { requested_key };
                    range = ReadRange {
                        byte_offset: bucket_byte_offset,
                        length: size_of::<BucketOffset>() as u64,
                    };
                }
            }
            let entry = Entry { meta, state };
            return Ok(Some((entry, range.clamp::<u8>(self.file_len))));
        }

        Ok(None)
    }

    /// Process a completed read result.
    fn process<E, F>(
        &mut self,
        entry: Entry<'key, Meta, K>,
        data: &[u8],
        f: &mut F,
    ) -> Result<(), E>
    where
        E: From<UniversalIoError>,
        F: FnMut(Meta, Option<PartialEntry<'_, K, V>>) -> Result<(), E>,
    {
        match entry.state {
            EntryState::ReadingOffset { requested_key } => {
                // Bucket-offset arrived: parse it, queue the entry for Loading.
                let entry_offset = parse_entry_offset(data)?;
                self.to_schedule.push(Entry {
                    meta: entry.meta,
                    state: EntryState::ReadingEntry {
                        byte_offset: self.map.entries_start + entry_offset,
                        buf: Vec::new(),
                        expected_len: self.entry_read_size_est,
                        requested_key: Some(requested_key),
                    },
                });
            }

            EntryState::ReadingEntry {
                byte_offset,
                mut buf,
                expected_len: _,
                mut requested_key,
            } => {
                buf.extend_from_slice(data);

                let parsed = PartialEntry::parse(&buf).unwrap(/* TODO */);

                if let Some(key) = requested_key {
                    if let Some(stored_key) = parsed.key() {
                        if key != stored_key {
                            f(entry.meta, None)?;
                            return Ok(());
                        }
                        requested_key = None;
                    }
                }

                if parsed.satisfies_kind(self.entry_kind) {
                    f(entry.meta, Some(parsed))?;
                } else {
                    self.to_schedule.push(Entry {
                        meta: entry.meta,
                        state: EntryState::ReadingEntry {
                            byte_offset,
                            // `+ 1` so the size strictly grows when `buf.len()` is already a
                            // power of two; otherwise the next refill reads 0 bytes and loops.
                            expected_len: (buf.len() as u64 + 1)
                                .next_power_of_two()
                                .max(K::VALUE_SIZE_EST as u64),
                            buf,
                            requested_key,
                        },
                    })
                }
            }
        }

        Ok(())
    }
}

struct Entry<'a, Meta, K: Key + ?Sized> {
    meta: Meta,
    state: EntryState<'a, K>,
}

// Lifecycle:
//   Request::Offset → Loading                  (offset already known)
//   Request::Key    → LocatingOffset → Loading (resolve via bucket pointer)
//   Loading         → Loading                  (with larger expected_len, on follow-up)
//   Loading         → done                     (try_read Ok or key mismatch)
enum EntryState<'a, K: Key + ?Sized> {
    // Waiting for the bucket-offset value (8 bytes) so we can locate the entry
    // data on disk. Only used for `Request::Key`.
    ReadingOffset {
        requested_key: &'a K,
    },
    // Reading the entry data. May span multiple I/Os if the entry is larger
    // than the initial size estimate.
    //
    // `requested_key` stays `Some` while we still need to verify the stored
    // key matches; gets cleared once verified, and is `None` for offset requests.
    ReadingEntry {
        byte_offset: u64,
        buf: Vec<u8>,
        expected_len: u64,
        requested_key: Option<&'a K>,
    },
}

enum Request<'a, K: Key + ?Sized> {
    Offset(u64),
    Key(&'a K),
}
