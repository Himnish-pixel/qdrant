// TODO: rename to "entry.rs"

use std::convert::Infallible;
use std::hash::Hash;
use std::io::Write;
use std::{io, str};

use zerocopy::{ConvertError, FromBytes, Immutable, IntoBytes, KnownLayout};

#[repr(C)]
#[derive(Copy, Clone, Debug, FromBytes, Immutable, IntoBytes, KnownLayout)]
pub(super) struct Header {
    pub key_type: [u8; 8],
    pub buckets_pos: u64,
    pub buckets_count: u64,
}

pub(super) type ValuesLen = u32;
pub(super) type BucketOffset = u64;

pub enum ReadError {
    Invalid,
    Incomplete,
}

pub type ReadResult<T> = Result<T, ReadError>;

/// A key that can be stored in the hash map.
pub trait Key: Sync + Hash {
    const ALIGN: usize;

    /// Stored in the file header.
    /// TODO: rename to `NAME_MAGIC`
    const NAME: [u8; 8];

    /// Reasonable guess size of the key when performing random read.
    /// Exact size for fixed-size keys, some arbitrary length for string keys.
    const VALUE_SIZE_EST: usize;

    /// Returns number of bytes which `write` will write.
    fn write_bytes(&self) -> usize;

    /// Write the key to `buf`.
    fn write(&self, buf: &mut impl Write) -> io::Result<()>;

    /// Check whether the first [`Key::write_bytes()`] of `buf` match the key.
    fn matches(&self, buf: &[u8]) -> bool;

    /// Try to read the key from `buf`.
    fn from_bytes(buf: &[u8]) -> Option<&Self>;

    /// Try to read the key from `buf`.
    ///
    /// This method can be called on an incomplete buffer. If it returns
    /// [`ReadError::Incomplete`], the caller will call this method again with
    /// new data appended to `buf`. New bytes are `buf[prev_size..]`. On first
    /// invocation, `prev_size` is 0.
    fn from_bytes_streaming(buf: &[u8], prev_size: usize) -> ReadResult<&Self>;
}

impl Key for str {
    const ALIGN: usize = align_of::<u8>();

    const VALUE_SIZE_EST: usize = 512;

    const NAME: [u8; 8] = *b"str\0\0\0\0\0";

    fn write_bytes(&self) -> usize {
        self.len() + 1
    }

    fn write(&self, buf: &mut impl Write) -> io::Result<()> {
        buf.write_all(self.as_bytes())?;
        buf.write_all(&[0xFF])?; // 0xFF is not a valid leading byte of a UTF-8 sequence.
        Ok(())
    }

    fn matches(&self, buf: &[u8]) -> bool {
        // The sentinel value 0xFF is used to ensure that `self` has the same length as the string
        // in the entry buffer.
        //
        // Suppose `self` is a prefix of the string in the entry buffer. (it's not very likely since
        // it would require a PHF collision, but it is still possible).
        // We'd like this method to return `false` in this case. So we need not just check that the
        // first `self.len()` bytes of `buf` are equal to `self`, but also that they have the same
        // length. To achieve that, we compare `self + [0xFF]` with `buf + [0xFF]`.
        //
        // ┌───self────┐       ┌───self────┐                 ┌─────self─────┐
        //  'f' 'o' 'o' FF      'f' 'o' 'o' FF                'f' 'o' 'o' FF
        //  'f' 'o' 'o' FF      'f' 'o' 'o' 'b' 'a' 'r' FF    'f' 'o' 'o' FF 'b' 'a' 'r' FF
        // └───entry───┘       └─────────entry─────────┘     └───────────entry──────────┘
        //    Case 1                    Case 2                          Case 3
        //    (happy)                 (collision)                   (never happens)
        //
        // 1. The case 1 is the happy path. This function returns `true`.
        // 2. In the case 2, `self` is a prefix of `entry`, but since we are also checking the
        //    sentinel, this function returns `false`. (0xFF != 'b')
        // 3. Hypothetical case 3 might never happen unless the index data is corrupted. This is
        //    because it assumes that `entry` is a concatenation of three parts: a valid UTF-8
        //    string ('foo'), a byte 0xFF, and the rest ('bar'). Concatenating a valid UTF-8 string
        //    with 0xFF will always result in an invalid UTF-8 string. Such string could not be
        //    added to the index since we are adding only valid UTF-8 strings as Rust enforces the
        //    validity of `str`/`String` types.
        buf.get(..self.len()) == Some(IntoBytes::as_bytes(self))
            && buf.get(self.len()) == Some(&0xFF)
    }

    fn from_bytes(buf: &[u8]) -> Option<&Self> {
        let len = buf.iter().position(|&b| b == 0xFF)?;
        str::from_utf8(&buf[..len]).ok()
    }

    fn from_bytes_streaming(buf: &[u8], prev_size: usize) -> ReadResult<&Self> {
        let Some(sentinel_pos) = buf.iter().skip(prev_size).position(|&b| b == 0xFF) else {
            return Err(ReadError::Incomplete);
        };
        str::from_utf8(&buf[..prev_size + sentinel_pos]).map_err(|_| ReadError::Invalid)
    }
}

impl Key for i64 {
    const ALIGN: usize = align_of::<i64>();

    const VALUE_SIZE_EST: usize = size_of::<i64>();

    const NAME: [u8; 8] = *b"i64\0\0\0\0\0";

    fn write_bytes(&self) -> usize {
        Self::VALUE_SIZE_EST
    }

    fn write(&self, buf: &mut impl Write) -> io::Result<()> {
        buf.write_all(self.as_bytes())
    }

    fn matches(&self, buf: &[u8]) -> bool {
        buf.get(..size_of::<i64>()) == Some(self.as_bytes())
    }

    fn from_bytes(buf: &[u8]) -> Option<&Self> {
        Some(i64::ref_from_prefix(buf).ok()?.0)
    }

    fn from_bytes_streaming(buf: &[u8], _prev_size: usize) -> ReadResult<&Self> {
        Ok(Self::ref_from_prefix(buf)?.0)
    }
}

impl Key for u128 {
    const ALIGN: usize = size_of::<u128>();

    const VALUE_SIZE_EST: usize = size_of::<u128>();

    const NAME: [u8; 8] = *b"u128\0\0\0\0";

    fn write_bytes(&self) -> usize {
        Self::VALUE_SIZE_EST
    }

    fn write(&self, buf: &mut impl Write) -> io::Result<()> {
        buf.write_all(self.as_bytes())
    }

    fn matches(&self, buf: &[u8]) -> bool {
        buf.get(..size_of::<u128>()) == Some(self.as_bytes())
    }

    fn from_bytes(buf: &[u8]) -> Option<&Self> {
        match u128::ref_from_prefix(buf) {
            Ok(res) => Some(res.0),
            Err(err) => {
                debug_assert!(false, "Error reading u128 from mmap: {err}");
                log::error!("Error reading u128 from mmap: {err}");
                None
            }
        }
    }

    fn from_bytes_streaming(buf: &[u8], _prev_size: usize) -> ReadResult<&Self> {
        Ok(Self::ref_from_prefix(buf)?.0)
    }
}

impl<A, S> From<ConvertError<A, S, Infallible>> for ReadError {
    fn from(err: ConvertError<A, S, Infallible>) -> ReadError {
        match err {
            ConvertError::Alignment(_) => ReadError::Invalid,
            ConvertError::Size(_) => ReadError::Incomplete,
        }
    }
}

pub enum ParsedEntry<'a, K: Key + ?Sized, V> {
    Invalid,
    NoKey,
    Key(&'a K),
    KeyAndValuesLen(&'a K, u32),
    KeyAndValues(&'a K, &'a [V], &'a [u8]),
}

pub fn parse_entry<'a, K: Key + ?Sized, V: FromBytes + Immutable>(
    buf: &'a [u8],
) -> ParsedEntry<'a, K, V> {
    // ┌─1:pad─┬────2:key─────┬─3:pad─┬─4:len─┬─5:pad─┬─────6:vals─────┐
    // │ · · · │ "abcdef\xFF" │ · · · │   5   │ · · · │ 10 20 30 40 50 │
    // └───────┴──────────────┴───────┴───────┴───────┴────────────────┘

    // 1. padding for the key
    let Some(buf) = align_slice_to(K::ALIGN, buf) else {
        return ParsedEntry::NoKey;
    };

    // 2. key
    let key = match K::from_bytes_streaming(buf, 0) {
        Ok(k) => k,
        Err(ReadError::Incomplete) => return ParsedEntry::NoKey,
        Err(ReadError::Invalid) => return ParsedEntry::Invalid,
    };
    let Some(buf) = buf.get(key.write_bytes()..) else {
        return ParsedEntry::Key(key);
    };

    // 3. padding for values_len
    let Some(buf) = align_slice_to(size_of::<ValuesLen>(), buf) else {
        return ParsedEntry::Key(key);
    };

    // 4. values_len
    let (&values_len, buf) = match ValuesLen::ref_from_prefix(buf) {
        Ok(v) => v,
        Err(ConvertError::Alignment(_)) => return ParsedEntry::Invalid,
        Err(ConvertError::Size(_)) => return ParsedEntry::Key(key),
    };

    // 5. padding for values
    let Some(buf) = align_slice_to(size_of::<V>(), buf) else {
        return ParsedEntry::KeyAndValuesLen(key, values_len);
    };

    // 6. values
    let (values, buf) = match <[V]>::ref_from_prefix_with_elems(buf, values_len as usize) {
        Ok(v) => v,
        Err(ConvertError::Alignment(_)) => return ParsedEntry::Invalid,
        Err(ConvertError::Size(_)) => return ParsedEntry::KeyAndValuesLen(key, values_len),
    };

    ParsedEntry::KeyAndValues(key, values, buf)
}

fn align_slice_to(alignment: usize, data: &[u8]) -> Option<&[u8]> {
    data.get(data.as_ptr().align_offset(alignment)..)
}
