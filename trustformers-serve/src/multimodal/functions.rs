//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {

    use super::super::types::*;
    #[tokio::test]
    async fn test_multimodal_service_creation() {
        let config = MultiModalConfig::default();
        let service = MultiModalService::new(config).expect("test operation should succeed");
        assert!(service.config.enabled);
    }
    #[test]
    fn test_image_metadata_creation() {
        let metadata = ImageMetadata {
            format: ImageFormat::JPEG,
            dimensions: (1920, 1080),
            file_size: 1024 * 1024,
            mime_type: "image/jpeg".to_string(),
            color_space: Some("sRGB".to_string()),
            has_alpha: false,
            exif_data: None,
        };
        assert_eq!(metadata.dimensions, (1920, 1080));
        assert_eq!(metadata.file_size, 1024 * 1024);
    }
    #[test]
    fn test_media_data_base64() {
        let data = MediaData::Base64 {
            data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                .to_string(),
        };
        match data {
            MediaData::Base64 { data: _ } => {},
            _ => panic!("Expected MediaData::Base64 variant"),
        }
    }
}
