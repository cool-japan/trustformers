//! Comprehensive tests for multimodal/types.rs

#[cfg(test)]
mod tests {
    use crate::multimodal::types::{
        AudioFormat, AudioMetadata, BoundingBox, ContentValidationConfig, DocumentFormat,
        DocumentMetadata, ImageFormat, ImageMetadata, ImageProcessingConfig, MediaData,
        MultiModalError, MultiModalInput, ProcessingPriority, ProcessingStatus, ProcessingStep,
        ProcessingStepType, QualityMetrics, ResourceUsage, ResultType, StorageBackend, VideoFormat,
        VideoMetadata,
    };
    use std::collections::HashMap;

    // --- BoundingBox tests ---

    #[test]
    fn test_bounding_box_fields() {
        let bbox = BoundingBox {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 50.0,
            class: "person".to_string(),
            confidence: 0.95,
        };
        assert_eq!(bbox.x, 10.0);
        assert_eq!(bbox.y, 20.0);
        assert_eq!(bbox.width, 100.0);
        assert_eq!(bbox.height, 50.0);
        assert_eq!(bbox.class, "person");
        assert!((bbox.confidence - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_bounding_box_area_calculation() {
        let bbox = BoundingBox {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 50.0,
            class: "car".to_string(),
            confidence: 0.8,
        };
        let area = bbox.width * bbox.height;
        assert!((area - 5000.0).abs() < 1e-6);
    }

    // --- QualityMetrics tests ---

    #[test]
    fn test_quality_metrics_fields() {
        let qm = QualityMetrics {
            quality_score: 0.85,
            blur_score: Some(0.1),
            noise_level: Some(0.05),
            brightness: Some(0.6),
            contrast: Some(0.7),
        };
        assert!((qm.quality_score - 0.85).abs() < 1e-6);
        assert!(qm.blur_score.is_some());
        assert!(qm.noise_level.is_some());
    }

    #[test]
    fn test_quality_metrics_optional_fields_none() {
        let qm = QualityMetrics {
            quality_score: 0.5,
            blur_score: None,
            noise_level: None,
            brightness: None,
            contrast: None,
        };
        assert!(qm.blur_score.is_none());
        assert!(qm.noise_level.is_none());
        assert!(qm.brightness.is_none());
    }

    // --- MediaData tests ---

    #[test]
    fn test_media_data_base64_variant() {
        let md = MediaData::Base64 {
            data: "SGVsbG8=".to_string(),
        };
        if let MediaData::Base64 { data } = &md {
            assert_eq!(data, "SGVsbG8=");
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_media_data_url_variant() {
        let md = MediaData::Url {
            url: "https://example.com/image.jpg".to_string(),
        };
        if let MediaData::Url { url } = &md {
            assert!(url.starts_with("https://"));
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_media_data_filepath_variant() {
        let md = MediaData::FilePath {
            path: "/tmp/image.png".to_string(),
        };
        if let MediaData::FilePath { path } = &md {
            assert!(path.ends_with(".png"));
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_media_data_binary_variant() {
        let data = vec![0x89u8, 0x50, 0x4e, 0x47]; // PNG magic bytes
        let md = MediaData::Binary { data: data.clone() };
        if let MediaData::Binary { data: d } = &md {
            assert_eq!(*d, data);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_media_data_clone() {
        let md = MediaData::Base64 {
            data: "test_data".to_string(),
        };
        let cloned = md.clone();
        if let (MediaData::Base64 { data: d1 }, MediaData::Base64 { data: d2 }) = (&md, &cloned) {
            assert_eq!(d1, d2);
        } else {
            panic!("cloned variant should match");
        }
    }

    // --- MultiModalInput tests ---

    #[test]
    fn test_multimodal_input_text_variant() {
        let input = MultiModalInput::Text {
            content: "Hello, world!".to_string(),
        };
        if let MultiModalInput::Text { content } = input {
            assert_eq!(content, "Hello, world!");
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_multimodal_input_image_variant() {
        let metadata = ImageMetadata {
            format: ImageFormat::JPEG,
            dimensions: (1920, 1080),
            file_size: 1024 * 100,
            mime_type: "image/jpeg".to_string(),
            color_space: Some("sRGB".to_string()),
            has_alpha: false,
            exif_data: None,
        };
        let input = MultiModalInput::Image {
            data: MediaData::Url {
                url: "https://example.com/img.jpg".to_string(),
            },
            metadata,
        };
        if let MultiModalInput::Image { metadata: meta, .. } = &input {
            assert_eq!(meta.dimensions, (1920, 1080));
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_multimodal_input_audio_variant() {
        let metadata = AudioMetadata {
            format: AudioFormat::MP3,
            duration_seconds: 180.0,
            sample_rate: 44100,
            bit_rate: 320,
            channels: 2,
            file_size: 1024 * 1024 * 5,
            mime_type: "audio/mpeg".to_string(),
            codec: Some("mp3".to_string()),
        };
        let input = MultiModalInput::Audio {
            data: MediaData::FilePath {
                path: "/tmp/audio.mp3".to_string(),
            },
            metadata,
        };
        if let MultiModalInput::Audio { metadata: meta, .. } = &input {
            assert_eq!(meta.sample_rate, 44100);
            assert_eq!(meta.channels, 2);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_multimodal_input_video_variant() {
        let metadata = VideoMetadata {
            format: VideoFormat::MP4,
            duration_seconds: 60.0,
            resolution: (1280, 720),
            fps: 30.0,
            file_size: 1024 * 1024 * 50,
            mime_type: "video/mp4".to_string(),
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            bit_rate: 2000,
        };
        let input = MultiModalInput::Video {
            data: MediaData::Url {
                url: "https://cdn.example.com/video.mp4".to_string(),
            },
            metadata,
        };
        if let MultiModalInput::Video { metadata: meta, .. } = &input {
            assert_eq!(meta.resolution, (1280, 720));
            assert_eq!(meta.fps, 30.0);
        } else {
            panic!("wrong variant");
        }
    }

    // --- ProcessingStatus tests ---

    #[test]
    fn test_processing_status_success_variant() {
        let status = ProcessingStatus::Success;
        assert!(matches!(status, ProcessingStatus::Success));
    }

    #[test]
    fn test_processing_status_partial_success() {
        let status = ProcessingStatus::PartialSuccess {
            errors: vec!["minor error".to_string()],
        };
        if let ProcessingStatus::PartialSuccess { errors } = &status {
            assert_eq!(errors.len(), 1);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_processing_status_failed() {
        let status = ProcessingStatus::Failed {
            error: "fatal error".to_string(),
        };
        if let ProcessingStatus::Failed { error } = &status {
            assert!(error.contains("fatal"));
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_processing_status_timeout() {
        let status = ProcessingStatus::Timeout;
        assert!(matches!(status, ProcessingStatus::Timeout));
    }

    // --- ResultType tests ---

    #[test]
    fn test_result_type_classification() {
        assert!(matches!(
            ResultType::Classification,
            ResultType::Classification
        ));
    }

    #[test]
    fn test_result_type_object_detection() {
        assert!(matches!(
            ResultType::ObjectDetection,
            ResultType::ObjectDetection
        ));
    }

    #[test]
    fn test_result_type_custom() {
        let rt = ResultType::Custom {
            type_name: "my_task".to_string(),
        };
        if let ResultType::Custom { type_name } = rt {
            assert_eq!(type_name, "my_task");
        } else {
            panic!("wrong variant");
        }
    }

    // --- ResourceUsage tests ---

    #[test]
    fn test_resource_usage_fields() {
        let usage = ResourceUsage {
            cpu_time_ms: 100,
            memory_bytes: 1024 * 1024,
            gpu_time_ms: Some(50),
            storage_bytes: 4096,
        };
        assert_eq!(usage.cpu_time_ms, 100);
        assert_eq!(usage.memory_bytes, 1024 * 1024);
        assert_eq!(usage.gpu_time_ms, Some(50));
    }

    #[test]
    fn test_resource_usage_no_gpu() {
        let usage = ResourceUsage {
            cpu_time_ms: 200,
            memory_bytes: 512 * 1024,
            gpu_time_ms: None,
            storage_bytes: 0,
        };
        assert!(usage.gpu_time_ms.is_none());
    }

    // --- MultiModalError tests ---

    #[test]
    fn test_multimodal_error_unsupported_format() {
        let err = MultiModalError::UnsupportedFormat {
            format: "tiff".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("tiff"));
    }

    #[test]
    fn test_multimodal_error_file_too_large() {
        let err = MultiModalError::FileTooLarge {
            size: 1024 * 1024 * 100,
            max_size: 1024 * 1024 * 10,
        };
        let msg = err.to_string();
        assert!(msg.contains("104857600") || msg.contains("bytes"));
    }

    #[test]
    fn test_multimodal_error_processing_failed() {
        let err = MultiModalError::ProcessingFailed {
            message: "GPU OOM".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("GPU OOM"));
    }

    #[test]
    fn test_multimodal_error_content_validation_failed() {
        let err = MultiModalError::ContentValidationFailed {
            reason: "inappropriate content detected".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("inappropriate"));
    }

    // --- ProcessingStep tests ---

    #[test]
    fn test_processing_step_resize() {
        let step = ProcessingStep {
            name: "resize-step".to_string(),
            step_type: ProcessingStepType::Resize,
            parameters: HashMap::new(),
        };
        assert_eq!(step.name, "resize-step");
        assert!(matches!(step.step_type, ProcessingStepType::Resize));
    }

    #[test]
    fn test_processing_step_custom_handler() {
        let step = ProcessingStep {
            name: "custom-step".to_string(),
            step_type: ProcessingStepType::Custom {
                handler: "my_handler".to_string(),
            },
            parameters: HashMap::new(),
        };
        if let ProcessingStepType::Custom { handler } = &step.step_type {
            assert_eq!(handler, "my_handler");
        } else {
            panic!("wrong step type");
        }
    }

    // --- ImageMetadata tests ---

    #[test]
    fn test_image_metadata_png_format() {
        let meta = ImageMetadata {
            format: ImageFormat::PNG,
            dimensions: (800, 600),
            file_size: 50000,
            mime_type: "image/png".to_string(),
            color_space: None,
            has_alpha: true,
            exif_data: None,
        };
        assert!(matches!(meta.format, ImageFormat::PNG));
        assert_eq!(meta.dimensions.0, 800);
        assert_eq!(meta.dimensions.1, 600);
        assert!(meta.has_alpha);
    }

    // --- ContentValidationConfig tests ---

    #[test]
    fn test_content_validation_config_default_like() {
        let config = ContentValidationConfig {
            enabled: true,
            scan_inappropriate_content: true,
            scan_malware: false,
            allowed_mime_types: vec!["image/jpeg".to_string(), "image/png".to_string()],
            safety_model: None,
            safety_threshold: 0.8,
            content_fingerprinting: false,
            block_duplicates: false,
        };
        assert!(config.enabled);
        assert_eq!(config.allowed_mime_types.len(), 2);
        assert!((config.safety_threshold - 0.8).abs() < 1e-6);
    }

    // --- ProcessingPriority tests ---

    #[test]
    fn test_processing_priority_variants() {
        assert!(matches!(ProcessingPriority::Low, ProcessingPriority::Low));
        assert!(matches!(
            ProcessingPriority::Normal,
            ProcessingPriority::Normal
        ));
        assert!(matches!(ProcessingPriority::High, ProcessingPriority::High));
        assert!(matches!(
            ProcessingPriority::Urgent,
            ProcessingPriority::Urgent
        ));
    }

    // --- StorageBackend tests ---

    #[test]
    fn test_storage_backend_local_variant() {
        let backend = StorageBackend::Local;
        assert!(matches!(backend, StorageBackend::Local));
    }

    #[test]
    fn test_storage_backend_s3_variant() {
        let backend = StorageBackend::S3 {
            bucket: "my-bucket".to_string(),
            region: "us-east-1".to_string(),
        };
        if let StorageBackend::S3 { bucket, region } = backend {
            assert_eq!(bucket, "my-bucket");
            assert_eq!(region, "us-east-1");
        } else {
            panic!("wrong variant");
        }
    }

    // --- ImageProcessingConfig tests ---

    #[test]
    fn test_image_processing_config_fields() {
        let config = ImageProcessingConfig {
            max_dimensions: (4096, 4096),
            auto_resize: true,
            resize_dimensions: Some((1280, 720)),
            quality: 85,
            auto_convert: false,
            target_format: None,
            strip_metadata: true,
            generate_thumbnails: false,
            thumbnail_size: (128, 128),
        };
        assert_eq!(config.max_dimensions, (4096, 4096));
        assert!(config.auto_resize);
        assert_eq!(config.quality, 85);
        assert!(config.strip_metadata);
    }

    // --- DocumentMetadata tests ---

    #[test]
    fn test_document_metadata_pdf_format() {
        let meta = DocumentMetadata {
            format: DocumentFormat::PDF,
            page_count: Some(10),
            file_size: 1024 * 500,
            mime_type: "application/pdf".to_string(),
            title: Some("Test Document".to_string()),
            author: Some("Alice".to_string()),
            created_date: None,
            modified_date: None,
            language: Some("en".to_string()),
            version: None,
            subject: None,
            keywords: vec!["test".to_string(), "document".to_string()],
            encrypted: false,
            text_confidence: Some(0.98),
            character_count: Some(5000),
            word_count: Some(800),
            line_count: Some(100),
        };
        assert!(matches!(meta.format, DocumentFormat::PDF));
        assert_eq!(meta.page_count, Some(10));
        assert!(!meta.encrypted);
        assert_eq!(meta.keywords.len(), 2);
    }
}
