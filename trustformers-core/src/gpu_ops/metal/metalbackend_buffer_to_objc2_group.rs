//! # MetalBackend - buffer_to_objc2_group Methods
//!
//! This module contains method implementations for `MetalBackend`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::common::*;
use super::metalbackend_type::MetalBackend;

impl MetalBackend {
    /// Convert metal-rs Buffer to objc2-metal ProtocolObject
    pub(crate) fn buffer_to_objc2(
        buffer: &Arc<Buffer>,
    ) -> Result<Retained<ProtocolObject<dyn ObjC2Buffer>>> {
        let buffer_ptr = ForeignType::as_ptr(buffer.as_ref()) as *mut objc2::runtime::AnyObject;
        let buffer_objc2: Retained<ProtocolObject<dyn ObjC2Buffer>> = unsafe {
            Retained::retain(buffer_ptr as *mut ProtocolObject<dyn ObjC2Buffer>).ok_or_else(
                || {
                    TrustformersError::hardware_error(
                        "Failed to convert metal-rs Buffer to objc2-metal",
                        "buffer_to_objc2",
                    )
                },
            )?
        };
        Ok(buffer_objc2)
    }
}
