use std::fmt::Debug;

/// Producer-consumer FIFO. Optimized to avoid copying data.
pub struct StreamBuffer<T> {
    buf: Vec<T>,
    start: usize,
}

#[derive(Debug)]
pub enum StreamBufferStep {
    /// Advance the FIFO by this amount of items.
    Consumed(usize),
    /// Indicate that the consumer wants a contiguous slice of at least this
    /// size. Don't advance the FIFO.
    WantContiguos(usize),
}

impl StreamBufferStep {
    pub fn not_enough_data() -> Self {
        // Pass a very big number. In practice, `process` will break the loop
        // at this point.
        Self::WantContiguos(usize::MAX)
    }
}

impl<T: Clone + Debug> StreamBuffer<T> {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            start: 0,
        }
    }

    /// Short: append `new_items` to the FIFO and call `consumer` in a loop.
    ///
    /// Long: but with zero-copy optimizations. Most of the new items would not
    /// be copied into the internal buffer. Instead, the `customer` will be
    /// served directly from `new_items`, unless it spans across more than one
    /// `new_items` chunks.
    ///
    /// This optimization imposes a bit awkward API: the `consumer` gets two
    /// slices. If you concatenate them, you get the whole FIFO. So, it's like
    /// [`std::collections::VecDeque::as_slices`].
    ///
    /// If the consumer needs a contiguous slice of a certain size, it can
    /// return [`StreamBufferStep::WantContiguos`] with the required size.
    /// This method will expand the first slice and call the consumer again.
    pub fn process(
        &mut self,
        mut new_items: &[T],
        mut consumer: impl FnMut(&[T], &[T]) -> StreamBufferStep,
    ) {
        // Step 1: Handle leftovers from previous iterations.
        loop {
            let buf_len = self.buf.len() - self.start;
            if buf_len == 0 {
                break;
            }
            match consumer(&self.buf[self.start..], new_items) {
                StreamBufferStep::Consumed(count) => {
                    self.start += count.min(buf_len);
                    new_items = &new_items[count.saturating_sub(buf_len)..];
                    break;
                }
                StreamBufferStep::WantContiguos(count) if count > buf_len + new_items.len() => {
                    // Not enough data in the FIFO at this moment.
                    self.extend(new_items);
                    return;
                }
                StreamBufferStep::WantContiguos(count) => {
                    let (part1, part2) = new_items.split_at(count.saturating_sub(buf_len));
                    self.extend(part1);
                    new_items = part2;
                    // We expect `Consumed(count)` on the next iteration, so
                    // this loop will take at most two iterations unless the
                    // `consumer` misbehaves.
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

        // Step 3: Prepare leftovers for the next iteration.
        self.extend(new_items);
    }

    fn extend(&mut self, items: &[T]) {
        if self.start == self.buf.len() {
            self.buf.clear();
            self.start = 0;
        } else if self.buf.len() + items.len() > self.buf.capacity() {
            self.buf.drain(..self.start);
            self.start = 0;
        }
        self.buf.extend_from_slice(items);
    }
}

#[cfg(test)]
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
