use std::io::{self, Cursor};
use std::marker::PhantomData;
use std::mem::size_of;
use std::path::Path;

use aligned_vec::AVec;
use ph::fmph::Function;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use super::bucket_offsets::BucketOffsets;
use crate::generic_consts::{Random, Sequential};
use crate::iterator_ext::ordering_iterator::OrderingIterator;
use crate::persisted_hashmap::keys::{Key, ReadError, ReadResult};
use crate::universal_io::{
    OpenOptions, ReadRange, Result, UniversalIoError, UniversalRead, UniversalReadPipeline,
};

type ValuesLen = u32;
type BucketOffset = u64;

/// Same on-disk layout as [`super::mmap_hashmap`]'s header.
#[repr(C)]
#[derive(Copy, Clone, Debug, FromBytes, Immutable, IntoBytes, KnownLayout)]
struct Header {
    key_type: [u8; 8],
    buckets_pos: u64,
    buckets_count: u64,
}

/// How many bytes a single lookup conceptually touches for hardware-counter accounting.
pub const READ_ENTRY_OVERHEAD: usize = super::mmap_hashmap::READ_ENTRY_OVERHEAD;

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
    _phantom_key: PhantomData<K>,
    _phantom_value: PhantomData<V>,
}

impl<
    K: Key + ?Sized,
    V: Sized + Copy + FromBytes + Immutable + IntoBytes + KnownLayout + bytemuck::Pod,
    R: UniversalRead<u8>,
> UniversalHashMap<K, V, R>
{
    const VALUES_LEN_SIZE: usize = size_of::<ValuesLen>();
    const VALUE_SIZE: usize = size_of::<V>();

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
            _phantom_key: PhantomData,
            _phantom_value: PhantomData,
        })
    }

    /// Number of distinct keys stored in the hash map.
    pub fn keys_count(&self) -> usize {
        self.header.buckets_count as usize
    }

    pub fn for_each_key<E: From<UniversalIoError>>(
        &self,
        mut f: impl FnMut(&K) -> Result<(), E>,
    ) -> Result<(), E>
    where
        K: PartialEq,
    {
        let offsets = self.read_all_bucket_offsets()?.to_sorted_vec();
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
        K: 'k + PartialEq,
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
        K: 'a + PartialEq,
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
        let file_len = UniversalRead::<u8>::len(&self.reader)?;

        let range = ReadRange {
            byte_offset: self.entries_start,
            length: file_len - self.entries_start,
        };

        let align = K::ALIGN.max(size_of::<ValuesLen>()).max(size_of::<V>());
        let mut buf = AVec::<u8>::new(align);
        let mut buf_pos = 0;

        let mut state = State { key_size: None };

        let iter = OrderingIterator::new(UniversalRead::<u8>::read_iter::<Sequential, usize>(
            &self.reader,
            range.iter_autochunks::<u8>().enumerate(),
        )?);

        for record in iter {
            let (_, mini_buf) = record?;
            buf.extend_from_slice(&mini_buf);

            let mut view: &[u8] = &buf;
            loop {
                match parse_entry(view, &mut state, buf_pos) {
                    Ok((data, key, values)) => {
                        f(key, values)?;
                        view = data;
                        state.key_size = None;
                        buf_pos = 0;
                    }
                    Err(ReadError::Incomplete) => {
                        buf_pos = view.len();
                        break;
                    }
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

    pub fn get(&self, key: &K) -> Result<Option<Vec<V>>>
    where
        K: PartialEq,
    {
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
    ///
    /// Only two IO reads are performed: bucket offset and entry header.
    pub fn get_values_count(&self, key: &K) -> Result<Option<usize>> {
        let Some((_entry_start, _header_size, values_len)) = self.lookup_entry_header(key)? else {
            return Ok(None);
        };
        Ok(Some(values_len as usize))
    }

    // ── Cache management ────────────────────────────────────────────────

    /// Populate the RAM cache for the backing file.
    pub fn populate(&self) -> Result<()> {
        UniversalRead::<u8>::populate(&self.reader)
    }

    /// Evict the backing file data from RAM cache.
    pub fn clear_ram_cache(&self) -> Result<()> {
        UniversalRead::<u8>::clear_ram_cache(&self.reader)
    }

    // ── Private helpers ─────────────────────────────────────────────────

    /// Hash `key`, read the bucket offset and entry header, verify the key matches,
    /// and return `(entry_start, header_size, values_len)`.
    ///
    /// Returns `Ok(None)` if the PHF has no mapping or the stored key doesn't match.
    /// Two IO reads are performed: bucket offset and entry header.
    fn lookup_entry_header(&self, key: &K) -> Result<Option<(u64, usize, u32)>> {
        let Some(hash) = self.phf.get(key) else {
            return Ok(None);
        };

        let entry_offset = self.read_bucket_offset(hash as usize)?;
        let entry_start = self.entries_start + entry_offset;
        let key_size_with_padding = Self::key_size_with_padding(key);
        let header_size = key_size_with_padding + Self::values_len_size_with_padding();

        let entry_header = self.reader.read::<Random>(ReadRange {
            byte_offset: entry_start,
            length: header_size as u64,
        })?;

        if !key.matches(&entry_header) {
            return Ok(None);
        }

        let values_len = Self::parse_values_len(&entry_header[key_size_with_padding..])?;
        Ok(Some((entry_start, header_size, values_len)))
    }

    /// Read all bucket offsets in one sequential IO.
    fn read_all_bucket_offsets(&self) -> Result<BucketOffsets<'_>> {
        let bucket_count = self.header.buckets_count as usize;
        let bytes = self.reader.read::<Sequential>(ReadRange {
            byte_offset: self.header.buckets_pos,
            length: (bucket_count * size_of::<BucketOffset>()) as u64,
        })?;
        Ok(BucketOffsets::new(bytes))
    }

    fn read_bucket_offset(&self, index: usize) -> Result<u64> {
        let byte_offset =
            self.header.buckets_pos + (index as u64) * size_of::<BucketOffset>() as u64;
        let bytes = self.reader.read::<Random>(ReadRange {
            byte_offset,
            length: size_of::<BucketOffset>() as u64,
        })?;
        let (offset, _) = BucketOffset::read_from_prefix(&bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Can't read bucket offset"))?;
        Ok(offset)
    }

    fn key_size_with_padding(key: &K) -> usize {
        key.write_bytes().next_multiple_of(Self::VALUE_SIZE)
    }

    const fn values_len_size_with_padding() -> usize {
        Self::VALUES_LEN_SIZE.next_multiple_of(Self::VALUE_SIZE)
    }

    fn parse_values_len(bytes: &[u8]) -> Result<u32> {
        let (len, _) = ValuesLen::read_from_prefix(bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Can't read values_len"))?;
        Ok(len)
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
        let file_len = UniversalRead::<u8>::len(&map.reader)?;
        Ok(Self {
            map,
            entry_kind,
            to_schedule: Vec::new(),
            entry_read_size_est,
            file_len,
        })
    }

    /// Produce the next entry to schedule, taking from `to_schedule` first and
    /// otherwise pulling a fresh `Request`. PHF misses are reported via
    /// `callback` and skipped over. Returns `Ok(None)` once both inputs are
    /// exhausted.
    fn refill<E, Cb>(
        &mut self,
        requests: &mut impl Iterator<Item = (Meta, Request<'k, K>)>,
        callback: &mut Cb,
    ) -> Result<Option<(Entry<'k, Meta, K>, ReadRange)>, E>
    where
        E: From<UniversalIoError>,
        Cb: FnMut(Meta, Option<PartialEntry<'_, K, V>>) -> Result<(), E>,
    {
        if let Some(entry) = self.to_schedule.pop() {
            let range = match &entry.state {
                EntryState::Loading {
                    byte_offset,
                    data_vec,
                    expected_len,
                    requested_key: _,
                } => {
                    let already = data_vec.len() as u64;
                    ReadRange {
                        byte_offset: *byte_offset + already,
                        length: *expected_len - already,
                    }
                }
                _ => unreachable!("only Loading entries are queued in to_schedule"),
            };
            return Ok(Some((entry, range.clamp::<u8>(self.file_len))));
        }
        // Pull from `requests`, skipping over PHF misses.
        while let Some((meta, request)) = requests.next() {
            let (entry, range) = match request {
                Request::Offset(offset) => {
                    // Offset request: location is known, jump straight to Loading.
                    let byte_offset = self.map.entries_start + offset;
                    (
                        Entry {
                            meta,
                            state: EntryState::Loading {
                                byte_offset,
                                data_vec: Vec::new(),
                                expected_len: self.entry_read_size_est,
                                requested_key: None,
                            },
                        },
                        ReadRange {
                            byte_offset,
                            length: self.entry_read_size_est,
                        },
                    )
                }
                Request::Key(key) => {
                    // PHF miss: no stored entry; report immediately and continue.
                    let Some(hash) = self.map.phf.get(key) else {
                        callback(meta, None)?;
                        continue;
                    };
                    // PHF hit: schedule the bucket-offset read; transitions to
                    // Loading once the offset arrives.
                    let bucket_byte_offset =
                        self.map.header.buckets_pos + hash * size_of::<BucketOffset>() as u64;
                    (
                        Entry {
                            meta,
                            state: EntryState::LocatingOffset { requested_key: key },
                        },
                        ReadRange {
                            byte_offset: bucket_byte_offset,
                            length: size_of::<BucketOffset>() as u64,
                        },
                    )
                }
            };
            return Ok(Some((entry, range.clamp::<u8>(self.file_len))));
        }
        Ok(None)
    }

    /// Process a completed read result. The caller is responsible for invoking
    /// `pipeline.wait()` and feeding `(entry, data)` back here.
    fn process<E, Cb>(
        &mut self,
        entry: Entry<'k, Meta, K>,
        data: &[u8],
        callback: &mut Cb,
    ) -> Result<(), E>
    where
        E: From<UniversalIoError>,
        Cb: FnMut(Meta, Option<PartialEntry<'_, K, V>>) -> Result<(), E>,
    {
        match entry.state {
            EntryState::LocatingOffset { requested_key } => {
                // Bucket-offset arrived: parse it, queue the entry for Loading.
                let entry_offset = parse_entry_offset(data)?;
                self.to_schedule.push(Entry {
                    meta: entry.meta,
                    state: EntryState::Loading {
                        byte_offset: self.map.entries_start + entry_offset,
                        data_vec: Vec::new(),
                        expected_len: self.entry_read_size_est,
                        requested_key: Some(requested_key),
                    },
                });
            }
            EntryState::Loading {
                byte_offset,
                mut data_vec,
                expected_len: _,
                requested_key,
            } => {
                data_vec.extend_from_slice(data);

                // For key requests, verify the stored key as soon as it's parseable.
                let mut unverified_key = requested_key;
                if let Some(req_key) = requested_key {
                    if let Some(stored_key) = K::from_bytes(&data_vec) {
                        if req_key != stored_key {
                            // Mismatch → no entry to return; skip further reads.
                            callback(entry.meta, None)?;
                            return Ok(());
                        }
                        // Match → no need to re-check on subsequent follow-up reads.
                        unverified_key = None;
                    }
                }

                match self.entry_kind.try_read::<K, V>(&data_vec) {
                    Ok(parsed) => callback(entry.meta, Some(parsed))?,
                    // Need more bytes: requeue with the new expected length.
                    Err(expected_len) => self.to_schedule.push(Entry {
                        meta: entry.meta,
                        state: EntryState::Loading {
                            byte_offset,
                            data_vec,
                            expected_len,
                            requested_key: unverified_key,
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
    LocatingOffset {
        requested_key: &'a K,
    },
    // Reading the entry data. May span multiple I/Os if the entry is larger
    // than the initial size estimate.
    //
    // `requested_key` stays `Some` while we still need to verify the stored
    // key matches; gets cleared once verified, and is `None` for offset requests.
    Loading {
        byte_offset: u64,
        data_vec: Vec<u8>,
        expected_len: u64,
        requested_key: Option<&'a K>,
    },
}

enum Request<'a, K: Key + ?Sized> {
    Offset(u64),
    Key(&'a K),
}

#[derive(Copy, Clone)]
enum PartialEntryKind {
    KeyOnly,
    KeyAndValues(u32),
}

impl PartialEntryKind {
    fn est_size<K: Key + ?Sized, V>(self) -> usize {
        match self {
            Self::KeyOnly => K::VALUE_SIZE_EST,
            Self::KeyAndValues(values_len) => {
                K::VALUE_SIZE_EST + size_of::<ValuesLen>() + size_of::<V>() * (values_len as usize)
            }
        }
    }

    fn try_read<K: Key + ?Sized, V: Sized + FromBytes + Immutable + IntoBytes + KnownLayout>(
        self,
        data: &[u8],
    ) -> Result<PartialEntry<'_, K, V>, u64> {
        let Some(key) = K::from_bytes(data) else {
            return Err(data.len().next_power_of_two() as u64);
        };
        if matches!(self, Self::KeyOnly) {
            return Ok(PartialEntry::KeyOnly(key));
        }

        let key_size = key.write_bytes();
        let values_len_start = key_size.next_multiple_of(size_of::<ValuesLen>());
        let values_len_end = values_len_start + size_of::<ValuesLen>();

        if data.len() < values_len_end {
            return Err(values_len_end as u64);
        }

        let values_len =
            ValuesLen::from_le_bytes(data[values_len_start..values_len_end].try_into().unwrap());

        let values_start = values_len_end.next_multiple_of(size_of::<V>());
        let values_end = values_start + values_len as usize * size_of::<V>();

        if data.len() < values_end {
            return Err(values_end as u64);
        }

        return Ok(PartialEntry::KeyAndValues(
            key,
            <[V]>::ref_from_bytes(&data[values_start..values_end]).unwrap(),
        ));
    }
}

enum PartialEntry<'a, K: Key + ?Sized, V> {
    KeyOnly(&'a K),
    KeyAndValues(&'a K, &'a [V]),
}

struct State {
    key_size: Option<usize>,
}

fn advance_slice_by(data: &[u8], by: usize) -> ReadResult<&[u8]> {
    data.get(by..).ok_or(ReadError::Incomplete)
}
fn advance_slice_to_align(align: usize, data: &[u8]) -> ReadResult<&[u8]> {
    advance_slice_by(data, data.as_ptr().align_offset(align))
}

fn parse_entry<'a, K: Key + ?Sized, V: Sized + FromBytes + Immutable + IntoBytes + KnownLayout>(
    buf: &'a [u8],
    state: &mut State,
    new_pos: usize,
) -> ReadResult<(&'a [u8], &'a K, &'a [V])> {
    // padding            padding                padding
    // ####### [ key... ] ####### [ values_len ] ####### [ values... ]
    // ^                  ^                      ^
    let data = advance_slice_to_align(K::ALIGN, buf)?;
    let entry_start_offset = buf.len() - data.len();

    let key_size;
    match state.key_size {
        Some(s) => key_size = s,
        None => {
            let key = K::from_bytes_streaming(data, new_pos.saturating_sub(entry_start_offset))?;
            key_size = key.write_bytes();
            state.key_size = Some(key_size);
        }
    }

    // The writer pads the key tail and the values_len tail to a multiple
    // of size_of::<V>() (not to the natural alignment of the trailing
    // field). Mirror that here.
    let v_size = size_of::<V>();
    let data = advance_slice_by(data, key_size.next_multiple_of(v_size))?;
    let (&values_len, data) = ValuesLen::ref_from_prefix(data)?;
    let values_len_pad = size_of::<ValuesLen>().next_multiple_of(v_size) - size_of::<ValuesLen>();
    let data = advance_slice_by(data, values_len_pad)?;
    let (values, data) = <[V]>::ref_from_prefix_with_elems(data, values_len as usize)?;

    let key = K::from_bytes(&buf[entry_start_offset..buf.len() - data.len()])
        .ok_or(ReadError::Invalid)?;

    Ok((data, key, values))
}
