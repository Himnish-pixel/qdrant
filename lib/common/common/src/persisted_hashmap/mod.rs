mod keys;
mod mmap_hashmap;
mod universal_hashmap;
mod write;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod universal_hashmap_tests;

pub use keys::Key;
pub use mmap_hashmap::{MmapHashMap, READ_ENTRY_OVERHEAD};
pub use universal_hashmap::UniversalHashMap;
pub use write::write;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

type ValuesLen = u32;
type BucketOffset = u64;

#[repr(C)]
#[derive(Copy, Clone, Debug, FromBytes, Immutable, IntoBytes, KnownLayout)]
struct Header {
    key_type: [u8; 8],
    buckets_pos: u64,
    buckets_count: u64,
}
