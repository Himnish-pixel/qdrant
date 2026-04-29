use std::collections::VecDeque;
use std::fmt::Debug;

use bytes::{Buf, BytesMut};

/// Producer-consumer buffer for streaming data.
///
/// Producer appends data to one side of FIFO buffer;
/// consumer consumes data from the other side.
///
/// Optimized to avoid allocations/copying:
struct StreamBuffer<T> {
    buf: Vec<T>,
    start: usize,
}

#[derive(Debug)]
pub enum StreamBufferStep {
    /// Advance the buffer by this amount of items.
    Consumed(usize),
    /// Request the buffer to be contiguous for the next `usize` items.
    /// Don't advance the buffer.
    WantContiguos(usize),
}

impl<T: Clone + Debug> StreamBuffer<T> {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            start: 0,
        }
    }

    fn process(
        &mut self,
        mut new_items: &[T],
        mut consumer: impl FnMut(&[T], &[T]) -> StreamBufferStep,
    ) {
        // Step 1: Handle leftovers from previous steps.
        // In practice, this loop will have at max 2 iterations:
        // - `WantContiguos(too big)` -> early return
        // - `WantContiguos(fits)`, then `Consumed` -> proceed to step 2.
        loop {
            let buf_len = self.buf.len() - self.start;
            if buf_len == 0 {
                break;
            }
            match consumer(&self.buf[self.start..], new_items) {
                StreamBufferStep::Consumed(count) => {
                    break;
                }
                StreamBufferStep::WantContiguos(count) if count > buf_len + new_items.len() => {
                    self.prepare_leftovers(new_items);
                    return;
                }
                StreamBufferStep::WantContiguos(count) => {
                    self.prepare_leftovers(&new_items[..count - buf_len]);
                    new_items = &new_items[count - buf_len..];
                }
            }
        }

        // Step 2: Main loop.
        // Consume `new_items` without copying them into `self.buf`.
        while new_items.len() != 0 {
            match consumer(new_items, &[]) {
                StreamBufferStep::Consumed(count) => new_items = &new_items[count..],
                StreamBufferStep::WantContiguos(_) => break,
            }
        }

        // Step 3: Prepare leftovers for the next step.
        self.prepare_leftovers(new_items);
    }

    fn prepare_leftovers(&mut self, items: &[T]) {
        if self.start == self.buf.len() {
            self.buf.clear();
            self.start = 0;
        } else if self.buf.len() + items.len() > self.buf.capacity() {
            // An optimization. The next `extend_from_slice` will reallocate and copy.
            self.buf.drain(..self.start);
            self.start = 0;
        }
        self.buf.extend_from_slice(items);
    }
}

mod tests {
    use itertools::assert_equal;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    #[test]
    fn test() {
        let mut rng = StdRng::seed_from_u64(0);

        // Test scenario:
        // - Producer produces chunks of random sizes.
        // - Consumer tries to consume contiguous chunks of a random size.
        //   If a chunk is not contiguous, consumers asks fo
        //   If not enough data or data is not contiguous, consumer not proceeds.

        for _ in 0..100 {
            let mut buf = StreamBuffer::new();

            let mut consumer_pos = 0;
            let mut consumer_target = rng.random_range(1..100);

            let mut producer_pos = 0;

            for _ in 0..20 {
                // Producer produces some data.
                let next_producer_pos = producer_pos + rng.random_range(1..100);
                let new_data = (producer_pos..next_producer_pos).collect::<Vec<_>>();
                producer_pos = next_producer_pos;

                buf.process(&new_data, |part1, part2| {
                    eprintln!("part1={:?}, part2={:?}", part1, part2);
                    // Check that the data is correct.
                    assert_equal(
                        part1.iter().chain(part2.iter()).copied(),
                        consumer_pos..consumer_pos + part1.len() + part2.len(),
                    );

                    let want_consume = consumer_target - consumer_pos;

                    let result;
                    if part1.len() >= want_consume {
                        // Chunk is contiguous, consume it.
                        result = StreamBufferStep::Consumed(want_consume);
                        // On next iteration, ask for next chunk.
                        consumer_pos = consumer_target;
                        consumer_target += rng.random_range(1..100);
                    } else {
                        // Chunk is not contiguous, ask for more data.
                        result = StreamBufferStep::WantContiguos(want_consume);
                    }
                    result
                });
            }
        }
    }
}
