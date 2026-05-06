use std::io::{self, Cursor};
use std::marker::PhantomData;
use std::mem::size_of;
use std::path::Path;

use aligned_vec::AVec;
use ph::fmph::Function;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use super::{BucketOffset, Header, ValuesLen};
use crate::generic_consts::{Random, Sequential};
use crate::iterator_ext::ordering_iterator::OrderingIterator;
use crate::persisted_hashmap::keys::{Key, ReadError, ReadResult};
use crate::universal_io::{
    OpenOptions, ReadRange, Result, UniversalIoError, UniversalRead, UniversalReadPipeline,
};

/// On-disk hash map accessed via [`UniversalRead`].
pub struct UniversalHashMap<
    K: ?Sized,
    V: Sized + FromBytes + Immutable + IntoBytes + KnownLayout + bytemuck::Pod,
    R: UniversalRead<u8>,
> {
    reader: R,
    header: Header,
    phf: Function,
    /// Absolute byte offset where entry data begins (right after the bucket offsets array).
    entries_start: u64,
    phantom: PhantomData<(V, K)>,
}

impl<
    K: Key + ?Sized + PartialEq,
    V: Sized + Copy + FromBytes + Immutable + IntoBytes + KnownLayout + bytemuck::Pod,
    R: UniversalRead<u8>,
> UniversalHashMap<K, V, R>
{
    /// Load the hash map from file.
    pub fn open(path: impl AsRef<Path>, options: OpenOptions) -> Result<Self> {
        let reader = R::open(path, options)?;

        // 1. Read header.
        let header_bytes = reader.read::<Sequential>(ReadRange {
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
        let phf_bytes = reader.read::<Sequential>(ReadRange {
            byte_offset: phf_region_start,
            length: phf_region_len,
        })?;
        let phf = Function::read(&mut Cursor::new(&*phf_bytes))?;

        let entries_start =
            header.buckets_pos + header.buckets_count * size_of::<BucketOffset>() as u64;

        Ok(Self {
            reader,
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
        let bytes = self.reader.read::<Sequential>(ReadRange {
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
                let Some(PartialEntry::KeyOnly(key)) = entry else {
                    unreachable!()
                };
                f(key)
            },
        )
    }

    pub fn batch_with_entry<'k, Meta, E: From<UniversalIoError>>(
        &self,
        keys: impl IntoIterator<Item = (Meta, &'k K)>,
        mut f: impl FnMut(Meta, Option<&[V]>) -> Result<(), E>,
    ) -> Result<(), E>
    where
        K: 'k,
    {
        self.for_each_sparse(
            PartialEntryKind::KeyAndValues(1),
            keys.into_iter()
                .map(|(meta, key)| (meta, Request::Key(key))),
            |meta, entry| {
                let values = entry.map(|e| {
                    let PartialEntry::KeyAndValues(_, values) = e else {
                        unreachable!()
                    };
                    values
                });
                f(meta, values)
            },
        )
    }

    fn for_each_sparse<'a, Meta, E: From<UniversalIoError>>(
        &self,
        entry_kind: PartialEntryKind,
        requests: impl Iterator<Item = (Meta, Request<'a, K>)>,
        mut f: impl FnMut(Meta, Option<PartialEntry<'_, K, V>>) -> Result<(), E>,
    ) -> Result<(), E>
    where
        K: 'a,
    {
        let mut sparse = SparsePipeline::new(self, entry_kind)?;
        let mut pipeline = R::ReadPipeline::<'_, Entry<'a, Meta, K>>::new()?;
        let mut requests = requests.into_iter();
        loop {
            while pipeline.can_schedule() {
                let Some((entry, range)) = sparse.refill(&mut requests, &mut f)? else {
                    break;
                };
                pipeline.schedule::<Random>(entry, &self.reader, range)?;
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
        let file_len = self.reader.len()?;

        let range = ReadRange {
            byte_offset: self.entries_start,
            length: file_len - self.entries_start,
        };

        let align = K::ALIGN.max(size_of::<ValuesLen>()).max(size_of::<V>());
        let mut buf = AVec::<u8>::new(align);
        let mut state = State::default();

        let iter = OrderingIterator::new(
            self.reader
                .read_iter::<Sequential, usize>(range.iter_autochunks::<u8>().enumerate())?,
        );

        for record in iter {
            let (_, mini_buf) = record?;
            buf.extend_from_slice(&mini_buf);

            let mut view: &[u8] = &buf;
            loop {
                match state.parse::<K, V>(view, PartialEntryKind::KeyAndValues(0)) {
                    Ok((PartialEntry::KeyAndValues(key, values), data)) => {
                        f(key, values)?;
                        view = data;
                        state.reset();
                    }
                    Ok(_) => unreachable!("requested KeyAndValues kind"),
                    Err(ReadError::Incomplete) => break,
                    Err(ReadError::Invalid) => {
                        return Err(uio_data_err("Failed to parse entry from bytes").into());
                    }
                }
            }

            let remaining = view.len();
            let consumed = buf.len() - remaining;
            buf.copy_within(consumed.., 0);
            buf.truncate(remaining);
        }

        if !buf.is_empty() {
            return Err(uio_data_err("Trailing bytes left after parsing all entries").into());
        }

        Ok(())
    }

    // ── Single-key lookup ───────────────────────────────────────────────

    pub fn get(&self, key: &K) -> Result<Option<Vec<V>>> {
        let mut result: Option<Vec<V>> = None;
        self.for_each_sparse(
            PartialEntryKind::KeyAndValues(1),
            std::iter::once(((), Request::Key(key))),
            |(), entry| -> Result<()> {
                result = entry.map(|e| {
                    let PartialEntry::KeyAndValues(_, values) = e else {
                        unreachable!()
                    };
                    values.to_vec()
                });
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
                result = entry.map(|e| {
                    let PartialEntry::KeyAndValuesLen(_, values_len) = e else {
                        unreachable!()
                    };
                    values_len as usize
                });
                Ok(())
            },
        )?;
        Ok(result)
    }

    // ── Cache management ────────────────────────────────────────────────

    /// Populate the RAM cache for the backing file.
    pub fn populate(&self) -> Result<()> {
        self.reader.populate()
    }

    /// Evict the backing file data from RAM cache.
    pub fn clear_ram_cache(&self) -> Result<()> {
        self.reader.clear_ram_cache()
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
/// [`refill`](Self::refill) — produce the next entry to schedule —
/// and [`process`](Self::process) — consume a completed read.
/// The caller drives the underlying I/O pipeline.
struct SparsePipeline<'m, 'k, Meta, K, V, R>
where
    K: Key + ?Sized + 'k,
    V: Sized + Copy + FromBytes + Immutable + IntoBytes + KnownLayout + bytemuck::Pod,
    R: UniversalRead<u8>,
{
    map: &'m UniversalHashMap<K, V, R>,
    entry_kind: PartialEntryKind,
    // Entries with a Loading-phase read ready to be scheduled. Filled when we
    // transition into Loading or need a follow-up read for partially-loaded data.
    to_schedule: Vec<Entry<'k, Meta, K>>,
    entry_read_size_est: u64,
    file_len: u64,
}

impl<'m, 'k, Meta, K, V, R> SparsePipeline<'m, 'k, Meta, K, V, R>
where
    K: Key + ?Sized + 'k + PartialEq,
    V: Sized + Copy + FromBytes + Immutable + IntoBytes + KnownLayout + bytemuck::Pod,
    R: UniversalRead<u8>,
{
    fn new(map: &'m UniversalHashMap<K, V, R>, entry_kind: PartialEntryKind) -> Result<Self> {
        let entry_read_size_est = entry_kind.est_size::<K, V>() as u64;
        let file_len = map.reader.len()?;
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
        requests: &mut impl Iterator<Item = (Meta, Request<'k, K>)>,
        f: &mut F,
    ) -> Result<Option<(Entry<'k, Meta, K>, ReadRange)>, E>
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
                    parse_state: _,
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
                        parse_state: State::default(),
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
    fn process<E, F>(&mut self, entry: Entry<'k, Meta, K>, data: &[u8], f: &mut F) -> Result<(), E>
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
                        parse_state: State::default(),
                    },
                });
            }

            EntryState::ReadingEntry {
                byte_offset,
                mut buf,
                expected_len: _,
                requested_key,
                mut parse_state,
            } => {
                buf.extend_from_slice(data);

                let parse_result = parse_state.parse::<K, V>(&buf, self.entry_kind);

                // Verify the requested key against the stored key as soon as
                // the key is parseable; bail early on mismatch.
                let mut unverified_key = requested_key;
                if let Some(req_key) = unverified_key {
                    if parse_state.key_size.is_some() {
                        let stored_key = parse_state
                            .read_key::<K>(&buf)
                            .map_err(|_| uio_data_err("Failed to read stored key"))?;
                        if req_key != stored_key {
                            f(entry.meta, None)?;
                            return Ok(());
                        }
                        unverified_key = None;
                    }
                }

                match parse_result {
                    Ok((parsed, _)) => f(entry.meta, Some(parsed))?,
                    Err(ReadError::Invalid) => {
                        return Err(uio_data_err("Failed to parse entry from bytes").into());
                    }
                    // Need more bytes: requeue with a heuristically grown length.
                    Err(ReadError::Incomplete) => self.to_schedule.push(Entry {
                        meta: entry.meta,
                        state: EntryState::ReadingEntry {
                            byte_offset,
                            expected_len: (buf.len() as u64)
                                .next_power_of_two()
                                .max(K::VALUE_SIZE_EST as u64),
                            buf,
                            requested_key: unverified_key,
                            parse_state,
                        },
                    }),
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
        parse_state: State,
    },
}

enum Request<'a, K: Key + ?Sized> {
    Offset(u64),
    Key(&'a K),
}

#[derive(Copy, Clone)]
enum PartialEntryKind {
    KeyOnly,
    KeyAndValuesLen,
    KeyAndValues(u32),
}

impl PartialEntryKind {
    fn est_size<K: Key + ?Sized, V>(self) -> usize {
        let v_size = size_of::<V>();
        // Upper bound: each component (key, values_len) may be followed by up to
        // `v_size - 1` bytes of padding before the next component.
        let key_padded = K::VALUE_SIZE_EST + v_size.saturating_sub(1);
        let values_len_padded = size_of::<ValuesLen>() + v_size.saturating_sub(1);
        match self {
            Self::KeyOnly => K::VALUE_SIZE_EST,
            Self::KeyAndValuesLen => key_padded + size_of::<ValuesLen>(),
            Self::KeyAndValues(values_len) => {
                key_padded + values_len_padded + v_size * (values_len as usize)
            }
        }
    }
}

enum PartialEntry<'a, K: Key + ?Sized, V> {
    KeyOnly(&'a K),
    KeyAndValuesLen(&'a K, u32),
    KeyAndValues(&'a K, &'a [V]),
}

/// Parsing state for one entry. Persists across calls to [`State::parse`] so
/// that the streaming key parse and the success-only [`State::read_key`] don't
/// re-scan bytes that have already been searched.
#[derive(Default)]
struct State {
    /// Size in bytes of the parsed key (set once the key is fully parsed).
    key_size: Option<usize>,
    /// `buf.len()` last seen. Tells [`Key::from_bytes_streaming`] how many
    /// leading bytes have already been searched for the key terminator.
    searched_up_to: usize,
}

impl State {
    fn reset(&mut self) {
        *self = Self::default();
    }

    /// Parse an entry from `buf` according to `kind`. Returns the parsed entry
    /// and the leftover slice on success. On `Incomplete`, mutates `self` so a
    /// follow-up call with a larger `buf` can resume without re-scanning.
    fn parse<'a, K: Key + ?Sized, V: Sized + FromBytes + Immutable + IntoBytes + KnownLayout>(
        &mut self,
        buf: &'a [u8],
        kind: PartialEntryKind,
    ) -> ReadResult<(PartialEntry<'a, K, V>, &'a [u8])> {
        // padding            padding                padding
        // ####### [ key... ] ####### [ values_len ] ####### [ values... ]
        // ^                  ^                      ^
        let entry_data = advance_slice_to_align(K::ALIGN, buf)?;
        let key_size = self.parse_key_size::<K>(buf, entry_data)?;

        if matches!(kind, PartialEntryKind::KeyOnly) {
            let remaining = advance_slice_by(entry_data, key_size)?;
            return Ok((PartialEntry::KeyOnly(self.read_key(entry_data)?), remaining));
        }

        // The writer pads the key tail and the values_len tail to a multiple
        // of size_of::<V>() (not to the natural alignment of the trailing
        // field). Mirror that here.
        let v_size = size_of::<V>();
        let data = advance_slice_by(entry_data, key_size.next_multiple_of(v_size))?;
        let (&values_len, data) = ValuesLen::ref_from_prefix(data)?;
        let values_len_pad =
            size_of::<ValuesLen>().next_multiple_of(v_size) - size_of::<ValuesLen>();
        let data = advance_slice_by(data, values_len_pad)?;

        if matches!(kind, PartialEntryKind::KeyAndValuesLen) {
            let entry = PartialEntry::KeyAndValuesLen(self.read_key(entry_data)?, values_len);
            return Ok((entry, data));
        }

        let (values, data) = <[V]>::ref_from_prefix_with_elems(data, values_len as usize)?;
        let entry = PartialEntry::KeyAndValues(self.read_key(entry_data)?, values);
        Ok((entry, data))
    }

    /// Read (or recall) the entry's key length, advancing the streaming key
    /// parse only if it hasn't already finished.
    fn parse_key_size<K: Key + ?Sized>(
        &mut self,
        buf: &[u8],
        entry_data: &[u8],
    ) -> ReadResult<usize> {
        if let Some(s) = self.key_size {
            return Ok(s);
        }
        let entry_start_offset = buf.len() - entry_data.len();
        let prev_size = self.searched_up_to.saturating_sub(entry_start_offset);
        match K::from_bytes_streaming(entry_data, prev_size) {
            Ok(key) => {
                let s = key.write_bytes();
                self.key_size = Some(s);
                Ok(s)
            }
            Err(e) => {
                self.searched_up_to = buf.len();
                Err(e)
            }
        }
    }

    /// Materialize a `&K` reference. Only call after [`Self::parse`] has
    /// populated `key_size` (i.e. on success paths or after a verify check).
    fn read_key<'a, K: Key + ?Sized>(&self, buf: &'a [u8]) -> ReadResult<&'a K> {
        let entry_data = advance_slice_to_align(K::ALIGN, buf)?;
        let key_size = self
            .key_size
            .expect("read_key called before the key was parsed");
        K::from_bytes(&entry_data[..key_size]).ok_or(ReadError::Invalid)
    }
}

fn advance_slice_by(data: &[u8], by: usize) -> ReadResult<&[u8]> {
    data.get(by..).ok_or(ReadError::Incomplete)
}
fn advance_slice_to_align(align: usize, data: &[u8]) -> ReadResult<&[u8]> {
    advance_slice_by(data, data.as_ptr().align_offset(align))
}
