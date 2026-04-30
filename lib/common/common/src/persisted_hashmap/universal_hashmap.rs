use std::borrow::Cow;
use std::io::{self, Cursor};
use std::marker::PhantomData;
use std::mem::size_of;
use std::path::Path;

use aligned_vec::AVec;
use itertools::{Either, Itertools};
use ph::fmph::Function;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use super::bucket_offsets::BucketOffsets;
use crate::generic_consts::{Random, Sequential};
use crate::iterator_ext::ordering_iterator::OrderingIterator;
use crate::persisted_hashmap::keys::{Key, ReadError, ReadResult};
use crate::universal_io::{OpenOptions, ReadRange, Result, UniversalIoError, UniversalRead};

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
///
/// Uses the same on-disk layout as [`super::mmap_hashmap::MmapHashMap`] and can open files
/// created by [`MmapHashMap::create`](super::mmap_hashmap::MmapHashMap::create).
///
/// Unlike the mmap variant, this implementation does not return borrowed slices into the
/// underlying storage. Instead it reads data on demand through the universal IO interface.
///
/// ## Access patterns
///
/// | Method              | IO reads | Allocations | Notes                              |
/// |---------------------|----------|-------------|------------------------------------|
/// | [`get_with`]        | 3        | 0–1         | Callback receives `&[V]` directly  |
/// | [`get`]             | 3        | 1           | Returns `Vec<V>`                   |
/// | [`get_values_count`]| 2        | 0           | Skips reading values entirely      |
/// | [`for_each_entry`]  | 2        | 0–N         | Bulk reads buckets + entries       |
///
/// [`get_with`]: Self::get_with
/// [`get`]: Self::get
/// [`get_values_count`]: Self::get_values_count
/// [`for_each_entry`]: Self::for_each_entry
pub struct UniversalHashMap<
    K: ?Sized,
    V: Sized + FromBytes + Immutable + IntoBytes + KnownLayout + bytemuck::Pod,
    R: UniversalRead<u8> + UniversalRead<V>,
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
    R: UniversalRead<u8> + UniversalRead<V>,
> UniversalHashMap<K, V, R>
{
    const VALUES_LEN_SIZE: usize = size_of::<ValuesLen>();
    const VALUE_SIZE: usize = size_of::<V>();

    /// Open the hash map from a file previously created by
    /// [`MmapHashMap::create`](super::mmap_hashmap::MmapHashMap::create).
    pub fn open(path: impl AsRef<Path>, options: OpenOptions) -> Result<Self> {
        let reader: R = UniversalRead::<u8>::open(path, options)?;

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

    pub fn for_each_key(&self, mut f: impl FnMut(&K) -> Result<()>) -> Result<()> {
        let offsets = self.read_all_bucket_offsets()?.to_sorted_vec();
        self.for_each_sparse_impl(PartialEntryKind::KeyOnly, offsets.into_iter(), |entry| {
            let PartialEntry::KeyOnly(key) = entry else {
                unreachable!()
            };
            f(key)
        })
    }

    // TODO: drop for_each_entry
    pub fn for_each_entry_v2(&self, mut f: impl FnMut(&K, &[V]) -> Result<()>) -> Result<()> {
        let offsets = self.read_all_bucket_offsets()?.to_sorted_vec();
        self.for_each_sparse_impl(
            PartialEntryKind::KeyAndValues(32), // TODO
            offsets.into_iter(),
            |entry| {
                let PartialEntry::KeyAndValues(key, values) = entry else {
                    unreachable!()
                };
                f(key, values)
            },
        )
    }

    fn for_each_sparse_impl(
        &self,
        entry_kind: PartialEntryKind,
        offsets: impl Iterator<Item = u64>,
        mut f: impl FnMut(PartialEntry<'_, K, V>) -> Result<()>,
    ) -> Result<()> {
        let expected_read_size = entry_kind.est_size::<K, V>() as u64;

        let file_len = UniversalRead::<u8>::len(&self.reader)?; // TODO: use .len()

        let reads = offsets.into_iter().map(|offset| {
            let byte_offset = self.entries_start + offset;
            let range = ReadRange {
                byte_offset,
                length: file_len.min(expected_read_size),
            };
            (byte_offset, range.clamp::<u8>(file_len))
        });

        struct ExtraRead {
            data_vec: Vec<u8>,
            byte_offset: u64,
            expected_len: u64,
        }
        let mut next_extra_reads = Vec::new();
        self.reader
            .read_batch::<Random, _>(reads, |byte_offset, data| {
                match entry_kind.try_read::<K, V>(data) {
                    Ok(entry) => f(entry),
                    Err(expected_len) => {
                        let mut data_vec = Vec::with_capacity(expected_len as usize);
                        data_vec.extend_from_slice(data);
                        next_extra_reads.push(ExtraRead {
                            data_vec,
                            byte_offset,
                            expected_len,
                        });
                        Ok(())
                    }
                }
            })?;

        while !next_extra_reads.is_empty() {
            let extra_reads = std::mem::take(&mut next_extra_reads)
                .into_iter()
                .map(|extra| {
                    let range = ReadRange {
                        byte_offset: extra.byte_offset + extra.data_vec.len() as u64,
                        length: file_len.min(extra.expected_len - extra.data_vec.len() as u64),
                    };
                    (extra, range.clamp::<u8>(file_len))
                });
            self.reader
                .read_batch::<Random, _>(extra_reads, |mut extra, data| {
                    extra.data_vec.extend_from_slice(data);
                    match entry_kind.try_read::<K, V>(&extra.data_vec) {
                        Ok(entry) => f(entry),
                        Err(expected_len) => {
                            extra.expected_len = expected_len;
                            next_extra_reads.push(extra);
                            Ok(())
                        }
                    }
                })?;
        }

        Ok(())
    }

    fn for_each_dense_impl(&self, mut f: impl FnMut(&K, &[V]) -> Result<()>) -> Result<()> {
        let file_len = UniversalRead::<u8>::len(&self.reader)?;

        let range = ReadRange {
            byte_offset: self.entries_start,
            length: file_len - self.entries_start,
        };

        let align = K::ALIGN.max(size_of::<ValuesLen>()).max(size_of::<V>());
        let mut buf = AVec::<u8>::new(align);
        let mut buf_pos = 0;

        let mut state = State { key_size: None };

        OrderingIterator::new(UniversalRead::<u8>::read_iter::<Sequential, usize>(
            &self.reader,
            range.iter_autochunks::<u8>().enumerate(),
        )?)
        .process_results(|it| -> Result<()> {
            for (_, mini_buf) in it {
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
                            return Err(uio_data_err("Failed to parse entry from bytes"));
                        }
                    }
                }

                let remaining = view.len();
                let consumed = buf.len() - remaining;
                buf.copy_within(consumed.., 0);
                buf.truncate(remaining);
            }
            Ok(())
        })??;

        if !buf.is_empty() {
            return Err(uio_data_err(
                "Trailing bytes left after parsing all entries",
            ));
        }

        Ok(())
    }

    // ── Single-key lookup ───────────────────────────────────────────────

    /// Look up the values associated with `key`, passing them to `f`.
    ///
    /// This is the most efficient lookup method: the callback receives a `&[V]` that may
    /// reference the backing storage directly (zero-copy for mmap-based readers) or a
    /// temporary read buffer.
    ///
    /// Three IO reads are performed: bucket offset, entry header, and values.
    pub fn get_with<T>(&self, key: &K, f: impl FnOnce(&[V]) -> T) -> Result<Option<T>> {
        let Some((entry_start, header_size, values_len)) = self.lookup_entry_header(key)? else {
            return Ok(None);
        };

        if values_len == 0 {
            return Ok(Some(f(&[])));
        }

        let values_start = entry_start + header_size as u64;
        let values_bytes = self.reader.read::<Random>(ReadRange {
            byte_offset: values_start,
            length: u64::from(values_len) * Self::VALUE_SIZE as u64,
        })?;

        Self::with_values(&values_bytes, f).map(Some)
    }

    pub fn get<'a>(
        &'a self,
        key: &K,
    ) -> Result<Option<impl Iterator<Item = Result<V>> + use<'a, K, V, R>>> {
        let Some((entry_start, header_size, values_len)) = self.lookup_entry_header(key)? else {
            return Ok(None);
        };

        if values_len == 0 {
            return Ok(Some(Either::Left(std::iter::empty())));
        }

        let range = ReadRange {
            byte_offset: entry_start + header_size as u64,
            length: u64::from(values_len),
        };

        let values = <R as UniversalRead<V>>::read::<Random>(&self.reader, range)?.into_owned();

        Ok(Some(Either::Right(values.into_iter().map(Ok))))
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

    // ── Batch lookup ────────────────────────────────────────────────────

    /// Look up multiple keys at once, returning results in the same order as the input.
    ///
    /// This is more efficient than calling [`get_with`](Self::get_with) in a loop because
    /// IO reads are sorted by file position and batched for sequential access.
    ///
    /// Three batched IO phases are performed:
    /// 1. Read bucket offsets (sorted by bucket index).
    /// 2. Read entry headers (sorted by entry offset) and verify keys.
    /// 3. Read values for matched entries (sorted by entry offset).
    pub fn get_with_batch<'k, T>(
        &self,
        keys: &[&'k K],
        mut f: impl FnMut(&K, &[V]) -> T,
    ) -> Result<Vec<Option<T>>>
    where
        K: 'k,
    {
        let (idx_mapping, bucket_ids): (Vec<_>, Vec<_>) = keys
            .iter()
            .enumerate()
            .filter_map(|(idx, key)| self.phf.get(key).map(|bucket_id| (idx, bucket_id)))
            .unzip();

        let entry_offsets = self.batch_resolve_bucket_offsets(bucket_ids)?;

        let (idx_mapping, values_offsets, values_lens) =
            self.batch_read_entry_headers(keys, idx_mapping, &entry_offsets)?;

        let results =
            self.batch_read_values(keys, idx_mapping, values_offsets, values_lens, &mut f)?;

        Ok(results)
    }

    // ── Iteration ───────────────────────────────────────────────────────

    /// Iterate over all entries, calling `f` for each `(key, values)` pair.
    ///
    /// Reads bucket offsets in bulk, sorts them for sequential access, then reads
    /// entries in batches of [`ENTRY_BATCH_SIZE`] to bound memory usage.
    pub fn for_each_entry(&self, mut f: impl FnMut(&K, &[V])) -> Result<()> {
        let bucket_count = self.header.buckets_count as usize;
        if bucket_count == 0 {
            return Ok(());
        }

        let buckets = self.read_all_bucket_offsets()?;
        let sorted_offsets = buckets.to_sorted_vec();

        let file_len = UniversalRead::<u8>::len(&self.reader)?;
        let entries_region_len = file_len - self.entries_start;

        const ENTRY_BATCH_SIZE: usize = 64;

        for chunk_start in (0..sorted_offsets.len()).step_by(ENTRY_BATCH_SIZE) {
            let chunk_end = (chunk_start + ENTRY_BATCH_SIZE).min(sorted_offsets.len());

            let range_start = sorted_offsets[chunk_start];
            let range_end = sorted_offsets
                .get(chunk_end)
                .copied()
                .unwrap_or(entries_region_len);

            let chunk_data = self.reader.read::<Sequential>(ReadRange {
                byte_offset: self.entries_start + range_start,
                length: range_end - range_start,
            })?;

            for &offset in &sorted_offsets[chunk_start..chunk_end] {
                let local_offset = (offset - range_start) as usize;
                let entry = chunk_data.get(local_offset..).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "Entry offset out of bounds")
                })?;

                let Some(key) = K::from_bytes(entry) else {
                    debug_assert!(false, "Error reading key");
                    log::error!("Error reading key");
                    continue;
                };

                let key_size_with_padding = Self::key_size_with_padding(key);
                let values_len = Self::parse_values_len(
                    entry.get(key_size_with_padding..).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "Entry too short for values_len")
                    })?,
                )?;

                let values_from = key_size_with_padding + Self::values_len_size_with_padding();
                let values_to = values_from + values_len as usize * Self::VALUE_SIZE;

                let values_bytes = entry.get(values_from..values_to).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "Values region out of bounds")
                })?;

                Self::with_values(values_bytes, |values| f(key, values))?;
            }
        }

        Ok(())
    }

    // ── Borrowed iteration ──────────────────────────────────────────────

    /// Iterate over all entries, returning owned `(key, values)` pairs.
    ///
    /// Reads the entire entries region at once. Works with both borrowed
    /// (mmap) and owned (io_uring, etc.) storage backends.
    pub fn iter(&self) -> Result<impl Iterator<Item = (<K as ToOwned>::Owned, Vec<V>)>>
    where
        K: ToOwned,
    {
        let offsets = self.read_all_bucket_offsets()?.to_sorted_vec();
        let data = self.read_entries_region()?;
        Ok(offsets.into_iter().filter_map(move |off| {
            let (key, values) = Self::parse_entry_ref(data.get(off as usize..)?)?;
            Some((key.to_owned(), values.to_vec()))
        }))
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

    fn read_entries_region(&self) -> Result<Cow<'_, [u8]>> {
        let len = UniversalRead::<u8>::len(&self.reader)? - self.entries_start;
        self.reader.read::<Sequential>(ReadRange {
            byte_offset: self.entries_start,
            length: len,
        })
    }

    fn parse_entry_ref<'a>(entry: &'a [u8]) -> Option<(&'a K, &'a [V])> {
        let key = K::from_bytes(entry)?;
        let kp = Self::key_size_with_padding(key);
        let (vl, _) = ValuesLen::read_from_prefix(entry.get(kp..)?).ok()?;
        let vf = kp + Self::values_len_size_with_padding();
        let vt = vf + vl as usize * Self::VALUE_SIZE;
        let values = <[V]>::ref_from_bytes(entry.get(vf..vt)?).ok()?;
        Some((key, values))
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

    /// Interpret `bytes` as `&[V]` and pass to the callback.
    ///
    /// Fast path: if `bytes` are properly aligned for `V`, zero-copy reinterpretation.
    /// Slow path: copy into an aligned `Vec<V>` element-by-element.
    fn with_values<T>(bytes: &[u8], f: impl FnOnce(&[V]) -> T) -> Result<T> {
        if let Ok(values) = <[V]>::ref_from_bytes(bytes) {
            return Ok(f(values));
        }
        debug_assert!(
            false,
            "Values bytes not aligned for zero-copy; falling back to copy"
        );
        let values = Self::copy_values_from_bytes(bytes)?;
        Ok(f(&values))
    }

    /// Copy bytes into a `Vec<V>` one element at a time (alignment-safe).
    fn copy_values_from_bytes(bytes: &[u8]) -> Result<Vec<V>> {
        if Self::VALUE_SIZE == 0 || !bytes.len().is_multiple_of(Self::VALUE_SIZE) {
            return Err(uio_data_err(format!(
                "Values byte length {} is not a multiple of value size {}",
                bytes.len(),
                Self::VALUE_SIZE,
            )));
        }
        bytes
            .chunks_exact(Self::VALUE_SIZE)
            .map(|chunk| {
                V::read_from_bytes(chunk).map_err(|_| uio_data_err("Can't read value from bytes"))
            })
            .collect()
    }

    // ── Batch-lookup helpers ───────────────────────────────────────────

    /// Phase 1: resolve bucket index → entry offset for each lookup.
    fn batch_resolve_bucket_offsets(&self, bucket_ids: Vec<u64>) -> Result<Vec<u64>> {
        let mut entry_offsets: Vec<u64> = vec![0; bucket_ids.len()];

        let ranges = bucket_ids.into_iter().enumerate().map(|(idx, bucket_idx)| {
            (
                idx,
                ReadRange {
                    byte_offset: self.header.buckets_pos
                        + bucket_idx * size_of::<BucketOffset>() as u64,
                    length: size_of::<BucketOffset>() as u64,
                },
            )
        });

        self.reader.read_batch::<Random, _>(ranges, |idx, data| {
            let (offset, _) =
                BucketOffset::read_from_prefix(data).map_err(|e| uio_data_err(e.to_string()))?;
            entry_offsets[idx] = offset;
            Ok(())
        })?;

        Ok(entry_offsets)
    }

    /// Phase 2: read entry headers, verify keys, and parse values_len.
    fn batch_read_entry_headers(
        &self,
        keys: &[&K],
        idx_mapping: Vec<usize>,
        entry_offsets: &[u64],
    ) -> Result<(Vec<usize>, Vec<u64>, Vec<u32>)> {
        let mut new_idx_mapping = Vec::with_capacity(entry_offsets.len());
        let mut values_offsets = Vec::with_capacity(entry_offsets.len());
        let mut values_lens = Vec::with_capacity(entry_offsets.len());

        let ranges = entry_offsets.iter().enumerate().map(|(idx, entry_offset)| {
            let key = keys[idx_mapping[idx]];
            let header_size =
                Self::key_size_with_padding(key) + Self::values_len_size_with_padding();
            (
                idx,
                ReadRange {
                    byte_offset: self.entries_start + *entry_offset,
                    length: header_size as u64,
                },
            )
        });

        self.reader.read_batch::<Random, _>(ranges, |idx, data| {
            let key_id = idx_mapping[idx];
            let key = keys[key_id];
            let header_size =
                Self::key_size_with_padding(key) + Self::values_len_size_with_padding();
            let entry_offset = entry_offsets[idx];

            if !key.matches(data) {
                return Ok(());
            }
            let key_pad = Self::key_size_with_padding(key);
            let vl_bytes = data
                .get(key_pad..)
                .ok_or_else(|| uio_data_err("Entry too short for values_len"))?;
            let (vl, _) =
                ValuesLen::read_from_prefix(vl_bytes).map_err(|e| uio_data_err(e.to_string()))?;

            let values_offset = self.entries_start + entry_offset + header_size as u64;

            values_offsets.push(values_offset);
            values_lens.push(vl);
            new_idx_mapping.push(key_id);
            Ok(())
        })?;

        Ok((new_idx_mapping, values_offsets, values_lens))
    }

    /// Phase 3: read values for matched entries and populate results.
    fn batch_read_values<T>(
        &self,
        keys: &[&K],
        idx_mapping: Vec<usize>,
        values_offsets: Vec<u64>,
        values_lens: Vec<u32>,
        f: &mut impl FnMut(&K, &[V]) -> T,
    ) -> Result<Vec<Option<T>>> {
        let mut results: Vec<Option<T>> = Vec::with_capacity(keys.len());
        results.resize_with(keys.len(), || None);

        // Handle zero-length value matches.
        for (idx, &values_len) in values_lens.iter().enumerate() {
            if values_len == 0 {
                let orig_idx = idx_mapping[idx];
                results[orig_idx] = Some(f(keys[orig_idx], &[]));
            }
        }

        if values_lens.is_empty() {
            return Ok(results);
        }

        let ranges = values_offsets
            .into_iter()
            .zip(values_lens)
            .map(|(values_offset, values_len)| ReadRange {
                byte_offset: values_offset,
                length: u64::from(values_len) * Self::VALUE_SIZE as u64,
            })
            .enumerate();

        self.reader
            .read_batch::<Sequential, _>(ranges, |idx, data| {
                let key_id = idx_mapping[idx];
                let key = keys[key_id];
                Self::with_values(data, |values| {
                    results[key_id] = Some(f(key, values));
                })
            })?;

        Ok(results)
    }
}

fn uio_data_err(msg: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> UniversalIoError {
    UniversalIoError::Io(io::Error::new(io::ErrorKind::InvalidData, msg))
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
