//! # BufferIterator - Trait Implementations
//!
//! This module contains trait implementations for `BufferIterator`.
//!
//! ## Implemented Traits
//!
//! - `Iterator`
//! - `ExactSizeIterator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::BufferIterator;

impl<'a, T> Iterator for BufferIterator<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let item = self.buffer.get(self.current_index);
        self.current_index += 1;
        self.remaining -= 1;
        item
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, T> ExactSizeIterator for BufferIterator<'a, T> {}
