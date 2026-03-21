//! # FilteredEventStream - Trait Implementations
//!
//! This module contains trait implementations for `FilteredEventStream`.
//!
//! ## Implemented Traits
//!
//! - `Stream`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio_stream::Stream;
use tracing::warn;

impl Stream for FilteredEventStream {
    type Item = StreamEvent;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match Pin::new(&mut self.receiver).poll_next(cx) {
                Poll::Ready(Some(Ok(event))) => {
                    if StreamSubscription::matches_filter(&event, &self.filter) {
                        return Poll::Ready(Some(event));
                    }
                },
                Poll::Ready(Some(Err(e))) => {
                    warn!("Stream error: {}", e);
                },
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}
