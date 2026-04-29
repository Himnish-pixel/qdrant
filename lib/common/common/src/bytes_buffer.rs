use std::collections::VecDeque;

use bytes::{Buf, BytesMut};

pub enum StreamBufferStep {
    Consumed(usize),
    WantContiguos(usize),
}

struct StreamBuffer<T> {
    buf: Vec<T>,
    start: usize,
}

impl<T: Clone> StreamBuffer<T> {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            start: 0,
        }
    }

    fn process(&mut self, new_data: &[T], mut f: impl FnMut(&[T], &[T]) -> StreamBufferStep) {
        let mut new_start = 0;

        loop {
            let buf_len = self.buf.len() - self.start;
            let new_len = new_data.len() - new_start;

            match f(&self.buf[self.start..], &new_data[new_start..]) {
                StreamBufferStep::Consumed(count) => {
                    let from_buf = count.min(buf_len);
                    self.start += from_buf;
                    new_start += count - from_buf;
                }
                StreamBufferStep::WantContiguos(count) => {
                    if buf_len >= count {
                        continue;
                    }
                    if buf_len + new_len < count {
                        break;
                    }
                    let need = count - buf_len;
                    self.append(&new_data[new_start..new_start + need]);
                    new_start += need;
                }
            }
        }

        if new_start < new_data.len() {
            self.append(&new_data[new_start..]);
        }
        if self.start == self.buf.len() {
            self.buf.clear();
            self.start = 0;
        }
    }

    fn append(&mut self, items: &[T]) {
        if self.start > 0 && self.buf.len() + items.len() > self.buf.capacity() {
            self.buf.drain(..self.start);
            self.start = 0;
        }
        self.buf.extend_from_slice(items);
    }
}

struct StreamBufferNaive<T> {
    buf: VecDeque<T>,
}

impl<T: Clone> StreamBufferNaive<T> {
    fn new() -> Self {
        Self {
            buf: VecDeque::new(),
        }
    }

    fn process(&mut self, new_data: &[T], mut f: impl FnMut(&[T], &[T]) -> StreamBufferStep) {
        self.buf.extend(new_data.iter().cloned());
        loop {
            let (a, b) = self.buf.as_slices();
            match f(a, b) {
                StreamBufferStep::Consumed(count) => {
                    self.buf.drain(..count);
                }
                StreamBufferStep::WantContiguos(count) => {
                    self.buf.make_contiguous();
                    if self.buf.len() < count {
                        break;
                    }
                }
            }
        }
    }
}
