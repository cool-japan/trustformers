//! Pose estimation pipeline — keypoint detection for human body (COCO-style 17 keypoints)

use std::fmt;

/// COCO keypoint type enumeration (17 canonical keypoints)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CocoKeypoint {
    Nose = 0,
    LeftEye = 1,
    RightEye = 2,
    LeftEar = 3,
    RightEar = 4,
    LeftShoulder = 5,
    RightShoulder = 6,
    LeftElbow = 7,
    RightElbow = 8,
    LeftWrist = 9,
    RightWrist = 10,
    LeftHip = 11,
    RightHip = 12,
    LeftKnee = 13,
    RightKnee = 14,
    LeftAnkle = 15,
    RightAnkle = 16,
}

impl CocoKeypoint {
    /// Human-readable name for this keypoint
    pub fn name(&self) -> &'static str {
        match self {
            CocoKeypoint::Nose => "nose",
            CocoKeypoint::LeftEye => "left_eye",
            CocoKeypoint::RightEye => "right_eye",
            CocoKeypoint::LeftEar => "left_ear",
            CocoKeypoint::RightEar => "right_ear",
            CocoKeypoint::LeftShoulder => "left_shoulder",
            CocoKeypoint::RightShoulder => "right_shoulder",
            CocoKeypoint::LeftElbow => "left_elbow",
            CocoKeypoint::RightElbow => "right_elbow",
            CocoKeypoint::LeftWrist => "left_wrist",
            CocoKeypoint::RightWrist => "right_wrist",
            CocoKeypoint::LeftHip => "left_hip",
            CocoKeypoint::RightHip => "right_hip",
            CocoKeypoint::LeftKnee => "left_knee",
            CocoKeypoint::RightKnee => "right_knee",
            CocoKeypoint::LeftAnkle => "left_ankle",
            CocoKeypoint::RightAnkle => "right_ankle",
        }
    }

    /// All 17 COCO keypoints in canonical order
    pub fn all() -> [CocoKeypoint; 17] {
        [
            CocoKeypoint::Nose,
            CocoKeypoint::LeftEye,
            CocoKeypoint::RightEye,
            CocoKeypoint::LeftEar,
            CocoKeypoint::RightEar,
            CocoKeypoint::LeftShoulder,
            CocoKeypoint::RightShoulder,
            CocoKeypoint::LeftElbow,
            CocoKeypoint::RightElbow,
            CocoKeypoint::LeftWrist,
            CocoKeypoint::RightWrist,
            CocoKeypoint::LeftHip,
            CocoKeypoint::RightHip,
            CocoKeypoint::LeftKnee,
            CocoKeypoint::RightKnee,
            CocoKeypoint::LeftAnkle,
            CocoKeypoint::RightAnkle,
        ]
    }

    /// True if this keypoint belongs to the left side of the body
    pub fn is_left_side(&self) -> bool {
        matches!(
            self,
            CocoKeypoint::LeftEye
                | CocoKeypoint::LeftEar
                | CocoKeypoint::LeftShoulder
                | CocoKeypoint::LeftElbow
                | CocoKeypoint::LeftWrist
                | CocoKeypoint::LeftHip
                | CocoKeypoint::LeftKnee
                | CocoKeypoint::LeftAnkle
        )
    }

    /// True if this keypoint belongs to the right side of the body
    pub fn is_right_side(&self) -> bool {
        matches!(
            self,
            CocoKeypoint::RightEye
                | CocoKeypoint::RightEar
                | CocoKeypoint::RightShoulder
                | CocoKeypoint::RightElbow
                | CocoKeypoint::RightWrist
                | CocoKeypoint::RightHip
                | CocoKeypoint::RightKnee
                | CocoKeypoint::RightAnkle
        )
    }
}

/// A single detected keypoint with position and confidence
#[derive(Debug, Clone)]
pub struct Keypoint {
    pub keypoint_type: CocoKeypoint,
    /// Normalized horizontal position in [0, 1]
    pub x: f32,
    /// Normalized vertical position in [0, 1] (increases downward)
    pub y: f32,
    /// Visibility/confidence score in [0, 1]
    pub confidence: f32,
    /// True when confidence exceeds the pipeline visibility threshold
    pub is_visible: bool,
}

impl Keypoint {
    /// Construct a keypoint; `is_visible` is set by comparing `confidence` to `threshold`
    pub fn new(kp_type: CocoKeypoint, x: f32, y: f32, confidence: f32, threshold: f32) -> Self {
        Self {
            keypoint_type: kp_type,
            x,
            y,
            confidence,
            is_visible: confidence > threshold,
        }
    }

    /// Euclidean distance to another keypoint in normalised image space
    pub fn distance_to(&self, other: &Keypoint) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// A directed edge in the body skeleton connecting two keypoints
#[derive(Debug, Clone, Copy)]
pub struct SkeletonEdge {
    pub from: CocoKeypoint,
    pub to: CocoKeypoint,
}

/// Standard COCO body skeleton — 19 edges covering major limbs and head connections
pub fn coco_skeleton() -> Vec<SkeletonEdge> {
    vec![
        SkeletonEdge {
            from: CocoKeypoint::Nose,
            to: CocoKeypoint::LeftEye,
        },
        SkeletonEdge {
            from: CocoKeypoint::Nose,
            to: CocoKeypoint::RightEye,
        },
        SkeletonEdge {
            from: CocoKeypoint::LeftEye,
            to: CocoKeypoint::LeftEar,
        },
        SkeletonEdge {
            from: CocoKeypoint::RightEye,
            to: CocoKeypoint::RightEar,
        },
        SkeletonEdge {
            from: CocoKeypoint::LeftShoulder,
            to: CocoKeypoint::RightShoulder,
        },
        SkeletonEdge {
            from: CocoKeypoint::LeftShoulder,
            to: CocoKeypoint::LeftElbow,
        },
        SkeletonEdge {
            from: CocoKeypoint::RightShoulder,
            to: CocoKeypoint::RightElbow,
        },
        SkeletonEdge {
            from: CocoKeypoint::LeftElbow,
            to: CocoKeypoint::LeftWrist,
        },
        SkeletonEdge {
            from: CocoKeypoint::RightElbow,
            to: CocoKeypoint::RightWrist,
        },
        SkeletonEdge {
            from: CocoKeypoint::LeftShoulder,
            to: CocoKeypoint::LeftHip,
        },
        SkeletonEdge {
            from: CocoKeypoint::RightShoulder,
            to: CocoKeypoint::RightHip,
        },
        SkeletonEdge {
            from: CocoKeypoint::LeftHip,
            to: CocoKeypoint::RightHip,
        },
        SkeletonEdge {
            from: CocoKeypoint::LeftHip,
            to: CocoKeypoint::LeftKnee,
        },
        SkeletonEdge {
            from: CocoKeypoint::RightHip,
            to: CocoKeypoint::RightKnee,
        },
        SkeletonEdge {
            from: CocoKeypoint::LeftKnee,
            to: CocoKeypoint::LeftAnkle,
        },
        SkeletonEdge {
            from: CocoKeypoint::RightKnee,
            to: CocoKeypoint::RightAnkle,
        },
        SkeletonEdge {
            from: CocoKeypoint::LeftEar,
            to: CocoKeypoint::LeftShoulder,
        },
        SkeletonEdge {
            from: CocoKeypoint::RightEar,
            to: CocoKeypoint::RightShoulder,
        },
        SkeletonEdge {
            from: CocoKeypoint::Nose,
            to: CocoKeypoint::LeftShoulder,
        },
    ]
}

/// A detected person with all 17 keypoints and derived pose metadata
#[derive(Debug, Clone)]
pub struct PersonPose {
    pub keypoints: Vec<Keypoint>,
    /// Tight bounding box `(x_min, y_min, x_max, y_max)` over visible keypoints
    pub bounding_box: (f32, f32, f32, f32),
    /// Mean confidence of visible keypoints
    pub pose_score: f32,
    /// Unique person identifier within the result set
    pub person_id: usize,
}

impl PersonPose {
    /// Construct a `PersonPose`; bounding box and pose score are computed automatically
    pub fn new(keypoints: Vec<Keypoint>, person_id: usize) -> Self {
        let bbox = Self::compute_bounding_box(&keypoints);
        let score = Self::compute_pose_score(&keypoints);
        Self {
            keypoints,
            bounding_box: bbox,
            pose_score: score,
            person_id,
        }
    }

    fn compute_bounding_box(keypoints: &[Keypoint]) -> (f32, f32, f32, f32) {
        let visible: Vec<&Keypoint> = keypoints.iter().filter(|k| k.is_visible).collect();
        if visible.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let x_min = visible.iter().map(|k| k.x).fold(f32::INFINITY, f32::min);
        let y_min = visible.iter().map(|k| k.y).fold(f32::INFINITY, f32::min);
        let x_max = visible.iter().map(|k| k.x).fold(f32::NEG_INFINITY, f32::max);
        let y_max = visible.iter().map(|k| k.y).fold(f32::NEG_INFINITY, f32::max);
        (x_min, y_min, x_max, y_max)
    }

    fn compute_pose_score(keypoints: &[Keypoint]) -> f32 {
        let visible: Vec<f32> =
            keypoints.iter().filter(|k| k.is_visible).map(|k| k.confidence).collect();
        if visible.is_empty() {
            return 0.0;
        }
        visible.iter().sum::<f32>() / visible.len() as f32
    }

    /// Retrieve a specific keypoint by type
    pub fn get_keypoint(&self, kp_type: CocoKeypoint) -> Option<&Keypoint> {
        self.keypoints.iter().find(|k| k.keypoint_type == kp_type)
    }

    /// All keypoints whose confidence exceeds the visibility threshold
    pub fn visible_keypoints(&self) -> Vec<&Keypoint> {
        self.keypoints.iter().filter(|k| k.is_visible).collect()
    }

    /// Distance between the left and right shoulders (returns None if either is invisible)
    pub fn torso_width(&self) -> Option<f32> {
        let ls = self.get_keypoint(CocoKeypoint::LeftShoulder)?;
        let rs = self.get_keypoint(CocoKeypoint::RightShoulder)?;
        if !ls.is_visible || !rs.is_visible {
            return None;
        }
        Some(ls.distance_to(rs))
    }

    /// Returns true when the nose is above the left hip (y increases downward)
    pub fn is_upright(&self) -> bool {
        let nose = self.get_keypoint(CocoKeypoint::Nose);
        let left_hip = self.get_keypoint(CocoKeypoint::LeftHip);
        match (nose, left_hip) {
            (Some(n), Some(h)) => n.y < h.y,
            _ => false,
        }
    }
}

/// Full pose estimation result for one image
#[derive(Debug, Clone)]
pub struct PoseEstimationResult {
    pub persons: Vec<PersonPose>,
    pub width: usize,
    pub height: usize,
}

impl PoseEstimationResult {
    /// Number of detected persons
    pub fn num_persons(&self) -> usize {
        self.persons.len()
    }

    /// Persons whose overall pose score meets the threshold
    pub fn filter_by_score(&self, min_score: f32) -> Vec<&PersonPose> {
        self.persons.iter().filter(|p| p.pose_score >= min_score).collect()
    }
}

/// Errors that can occur during pose estimation
#[derive(Debug)]
pub enum PoseEstimationError {
    /// Image width or height is zero, or buffer length is inconsistent
    InvalidImageDimensions,
    /// The image buffer contains no data
    EmptyImage,
}

impl fmt::Display for PoseEstimationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PoseEstimationError::InvalidImageDimensions => {
                write!(f, "pose estimation error: invalid image dimensions")
            },
            PoseEstimationError::EmptyImage => {
                write!(f, "pose estimation error: image buffer is empty")
            },
        }
    }
}

impl std::error::Error for PoseEstimationError {}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Pose estimation pipeline compatible with ViTPose / HRNet output format
pub struct PoseEstimationPipeline {
    pub model: String,
    /// Confidence threshold for marking a keypoint as visible
    pub visibility_threshold: f32,
    /// Minimum pose score; persons below this are excluded from results
    pub min_pose_score: f32,
}

impl PoseEstimationPipeline {
    /// Create a pipeline with sensible defaults
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            visibility_threshold: 0.3,
            min_pose_score: 0.2,
        }
    }

    /// Estimate pose for all persons in the image.
    ///
    /// `image` is row-major `[H * W * 3]` f32 in `[0, 1]`.
    pub fn run(
        &self,
        image: &[f32],
        width: usize,
        height: usize,
    ) -> Result<PoseEstimationResult, PoseEstimationError> {
        if image.is_empty() {
            return Err(PoseEstimationError::EmptyImage);
        }
        if width == 0 || height == 0 || image.len() != width * height * 3 {
            return Err(PoseEstimationError::InvalidImageDimensions);
        }

        // Compute a simple deterministic image statistic to vary keypoint positions
        let pixel_mean: f32 = image.iter().sum::<f32>() / image.len() as f32;

        let person = self.build_mock_person(width, height, pixel_mean, 0, (0.0, 0.0, 1.0, 1.0));
        let persons =
            if person.pose_score >= self.min_pose_score { vec![person] } else { Vec::new() };

        Ok(PoseEstimationResult {
            persons,
            width,
            height,
        })
    }

    /// Estimate pose using provided person bounding box hints; one `PersonPose` per box.
    pub fn run_with_boxes(
        &self,
        image: &[f32],
        width: usize,
        height: usize,
        person_boxes: &[(f32, f32, f32, f32)],
    ) -> Result<PoseEstimationResult, PoseEstimationError> {
        if image.is_empty() {
            return Err(PoseEstimationError::EmptyImage);
        }
        if width == 0 || height == 0 || image.len() != width * height * 3 {
            return Err(PoseEstimationError::InvalidImageDimensions);
        }

        let pixel_mean: f32 = image.iter().sum::<f32>() / image.len() as f32;

        let mut persons = Vec::with_capacity(person_boxes.len());
        for (idx, &bbox) in person_boxes.iter().enumerate() {
            // Each box slightly shifts the mean for deterministic variety
            let local_mean = (pixel_mean + idx as f32 * 0.02).clamp(0.0, 1.0);
            let person = self.build_mock_person(width, height, local_mean, idx, bbox);
            if person.pose_score >= self.min_pose_score {
                persons.push(person);
            }
        }

        Ok(PoseEstimationResult {
            persons,
            width,
            height,
        })
    }

    /// Construct a deterministic mock `PersonPose` positioned within `bbox`
    fn build_mock_person(
        &self,
        _width: usize,
        _height: usize,
        image_mean: f32,
        person_id: usize,
        bbox: (f32, f32, f32, f32),
    ) -> PersonPose {
        let (bx_min, by_min, bx_max, by_max) = bbox;
        let bw = bx_max - bx_min;
        let bh = by_max - by_min;

        // Canonical relative keypoint positions within a unit bounding box
        // (x_rel, y_rel) — y=0 is top of bbox
        let canonical: [(f32, f32); 17] = [
            (0.50, 0.08), // Nose
            (0.42, 0.05), // LeftEye
            (0.58, 0.05), // RightEye
            (0.35, 0.07), // LeftEar
            (0.65, 0.07), // RightEar
            (0.30, 0.25), // LeftShoulder
            (0.70, 0.25), // RightShoulder
            (0.20, 0.45), // LeftElbow
            (0.80, 0.45), // RightElbow
            (0.15, 0.65), // LeftWrist
            (0.85, 0.65), // RightWrist
            (0.35, 0.55), // LeftHip
            (0.65, 0.55), // RightHip
            (0.30, 0.75), // LeftKnee
            (0.70, 0.75), // RightKnee
            (0.30, 0.95), // LeftAnkle
            (0.70, 0.95), // RightAnkle
        ];

        let all_types = CocoKeypoint::all();
        let keypoints: Vec<Keypoint> = all_types
            .iter()
            .zip(canonical.iter())
            .map(|(kp_type, (rx, ry))| {
                // Map into bbox, add tiny person-id perturbation for variety
                let x = bx_min + rx * bw + person_id as f32 * 0.005;
                let y = by_min + ry * bh + person_id as f32 * 0.005;
                // Confidence varies with image_mean and keypoint index to be deterministic
                let confidence = (image_mean * 0.5 + 0.5 + *rx * 0.1 - *ry * 0.05).clamp(0.0, 1.0);
                Keypoint::new(
                    *kp_type,
                    x.clamp(0.0, 1.0),
                    y.clamp(0.0, 1.0),
                    confidence,
                    self.visibility_threshold,
                )
            })
            .collect();

        PersonPose::new(keypoints, person_id)
    }
}

// ---------------------------------------------------------------------------
// Extended types and post-processor
// ---------------------------------------------------------------------------

/// Visibility state of a keypoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeypointVisibility {
    Visible,
    Occluded,
    NotLabeled,
}

/// A keypoint with a name, position, confidence, and visibility state.
#[derive(Debug, Clone)]
pub struct NamedKeypoint {
    pub name: String,
    pub x: f32,
    pub y: f32,
    pub confidence: f32,
    pub visibility: KeypointVisibility,
}

impl NamedKeypoint {
    pub fn new(name: &str, x: f32, y: f32, confidence: f32, threshold: f32) -> Self {
        let visibility = if confidence > threshold {
            KeypointVisibility::Visible
        } else if confidence > 0.0 {
            KeypointVisibility::Occluded
        } else {
            KeypointVisibility::NotLabeled
        };
        Self {
            name: name.to_string(),
            x,
            y,
            confidence,
            visibility,
        }
    }
}

/// A skeleton containing keypoints and connectivity edges.
#[derive(Debug, Clone)]
pub struct Skeleton {
    pub keypoints: Vec<NamedKeypoint>,
    pub connections: Vec<(usize, usize)>,
}

/// A full pose result with skeleton, bounding box, and score.
#[derive(Debug, Clone)]
pub struct PoseResult {
    pub skeleton: Skeleton,
    /// Bounding box `(x1, y1, x2, y2)` in normalised coords
    pub bbox: (f32, f32, f32, f32),
    pub score: f32,
}

/// Pose post-processing utilities.
pub struct PosePostprocessor;

impl PosePostprocessor {
    /// Build the standard COCO 17-keypoint skeleton with no detections.
    pub fn coco_skeleton() -> Skeleton {
        let names = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ];
        let keypoints = names
            .iter()
            .map(|&n| NamedKeypoint {
                name: n.to_string(),
                x: 0.0,
                y: 0.0,
                confidence: 0.0,
                visibility: KeypointVisibility::NotLabeled,
            })
            .collect();

        // Standard COCO skeleton connections (19 edges)
        let connections: Vec<(usize, usize)> = vec![
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4), // head
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10), // arms
            (5, 11),
            (6, 12),
            (11, 12), // torso
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16), // legs
            (3, 5),
            (4, 6),
            (0, 5), // ear-shoulder, nose-shoulder
        ];

        Skeleton {
            keypoints,
            connections,
        }
    }

    /// Find the peak (argmax) position in each 2D heatmap.
    ///
    /// Each heatmap is `Vec<Vec<f32>>` of shape `[H][W]`.
    /// Returns `(x, y, confidence)` per heatmap, normalised to `[0, 1]`.
    pub fn heatmap_to_keypoints(heatmaps: &[Vec<Vec<f32>>]) -> Vec<(f32, f32, f32)> {
        heatmaps
            .iter()
            .map(|hm| {
                if hm.is_empty() || hm[0].is_empty() {
                    return (0.0_f32, 0.0_f32, 0.0_f32);
                }
                let h = hm.len();
                let w = hm[0].len();
                let mut best_val = f32::NEG_INFINITY;
                let mut best_row = 0;
                let mut best_col = 0;
                for (r, row) in hm.iter().enumerate() {
                    for (c, &val) in row.iter().enumerate() {
                        if val > best_val {
                            best_val = val;
                            best_row = r;
                            best_col = c;
                        }
                    }
                }
                let x = best_col as f32 / (w - 1).max(1) as f32;
                let y = best_row as f32 / (h - 1).max(1) as f32;
                (x, y, best_val.clamp(0.0, 1.0))
            })
            .collect()
    }

    /// Normalise keypoint coordinates in place to `[0, 1]` given image dimensions.
    pub fn normalize_keypoints(keypoints: &mut Vec<NamedKeypoint>, image_w: f32, image_h: f32) {
        if image_w <= 0.0 || image_h <= 0.0 {
            return;
        }
        for kp in keypoints.iter_mut() {
            kp.x = (kp.x / image_w).clamp(0.0, 1.0);
            kp.y = (kp.y / image_h).clamp(0.0, 1.0);
        }
    }

    /// Object Keypoint Similarity (OKS) between predicted and ground-truth keypoints.
    ///
    /// `sigmas` are per-keypoint scale factors (standard COCO sigmas if available).
    pub fn oks_score(pred: &[NamedKeypoint], gt: &[NamedKeypoint], sigmas: &[f32]) -> f32 {
        if pred.is_empty() || gt.is_empty() {
            return 0.0;
        }
        let n = pred.len().min(gt.len()).min(sigmas.len());
        if n == 0 {
            return 0.0;
        }

        // Scale is set to 1.0 (unit image area) for normalised keypoints
        let scale = 1.0_f32;
        let mut sum = 0.0_f32;
        let mut count = 0_u32;

        for i in 0..n {
            if gt[i].visibility == KeypointVisibility::NotLabeled {
                continue;
            }
            let dx = pred[i].x - gt[i].x;
            let dy = pred[i].y - gt[i].y;
            let dist_sq = dx * dx + dy * dy;
            let s = 2.0 * sigmas[i] * sigmas[i] * scale;
            sum += (-dist_sq / s).exp();
            count += 1;
        }

        if count == 0 {
            0.0
        } else {
            sum / count as f32
        }
    }

    /// Percentage of Correct Keypoints (PCK): fraction of keypoints within
    /// `threshold` Euclidean distance of the ground truth.
    pub fn pck_accuracy(pred: &[NamedKeypoint], gt: &[NamedKeypoint], threshold: f32) -> f32 {
        if pred.is_empty() || gt.is_empty() {
            return 0.0;
        }
        let n = pred.len().min(gt.len());
        let correct: usize = (0..n)
            .filter(|&i| {
                if gt[i].visibility == KeypointVisibility::NotLabeled {
                    return false;
                }
                let dx = pred[i].x - gt[i].x;
                let dy = pred[i].y - gt[i].y;
                (dx * dx + dy * dy).sqrt() <= threshold
            })
            .count();

        let labeled: usize =
            (0..n).filter(|&i| gt[i].visibility != KeypointVisibility::NotLabeled).count();

        if labeled == 0 {
            0.0
        } else {
            correct as f32 / labeled as f32
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_image(w: usize, h: usize) -> Vec<f32> {
        let len = w * h * 3;
        (0..len).map(|i| (i as f32 / len as f32).clamp(0.0, 1.0)).collect()
    }

    #[test]
    fn test_coco_keypoint_name() {
        assert_eq!(CocoKeypoint::Nose.name(), "nose");
        assert_eq!(CocoKeypoint::LeftAnkle.name(), "left_ankle");
        assert_eq!(CocoKeypoint::RightWrist.name(), "right_wrist");
    }

    #[test]
    fn test_coco_keypoint_all_count() {
        assert_eq!(CocoKeypoint::all().len(), 17);
    }

    #[test]
    fn test_coco_keypoint_left_right_sides() {
        assert!(CocoKeypoint::LeftEye.is_left_side());
        assert!(!CocoKeypoint::LeftEye.is_right_side());
        assert!(CocoKeypoint::RightHip.is_right_side());
        assert!(!CocoKeypoint::RightHip.is_left_side());
        // Nose is neither left nor right
        assert!(!CocoKeypoint::Nose.is_left_side());
        assert!(!CocoKeypoint::Nose.is_right_side());
    }

    #[test]
    fn test_keypoint_visibility_threshold() {
        let kp_visible = Keypoint::new(CocoKeypoint::Nose, 0.5, 0.5, 0.8, 0.3);
        assert!(kp_visible.is_visible);

        let kp_invisible = Keypoint::new(CocoKeypoint::LeftEye, 0.5, 0.5, 0.1, 0.3);
        assert!(!kp_invisible.is_visible);

        // Exactly at threshold is NOT visible (confidence must be strictly greater)
        let kp_at_threshold = Keypoint::new(CocoKeypoint::RightEar, 0.5, 0.5, 0.3, 0.3);
        assert!(!kp_at_threshold.is_visible);
    }

    #[test]
    fn test_keypoint_distance() {
        let kp1 = Keypoint::new(CocoKeypoint::LeftShoulder, 0.0, 0.0, 0.9, 0.3);
        let kp2 = Keypoint::new(CocoKeypoint::RightShoulder, 0.3, 0.4, 0.9, 0.3);
        let dist = kp1.distance_to(&kp2);
        assert!((dist - 0.5).abs() < 1e-5, "expected 0.5, got {}", dist);
    }

    #[test]
    fn test_coco_skeleton_count() {
        assert_eq!(coco_skeleton().len(), 19);
    }

    #[test]
    fn test_person_pose_new() {
        let kps: Vec<Keypoint> = CocoKeypoint::all()
            .iter()
            .map(|kp| Keypoint::new(*kp, 0.5, 0.5, 0.9, 0.3))
            .collect();
        let person = PersonPose::new(kps, 0);
        assert_eq!(person.person_id, 0);
        assert_eq!(person.keypoints.len(), 17);
        assert!(person.pose_score > 0.0);
    }

    #[test]
    fn test_person_pose_bounding_box() {
        // Two visible keypoints at different positions
        let mut kps: Vec<Keypoint> = Vec::new();
        kps.push(Keypoint::new(CocoKeypoint::Nose, 0.2, 0.1, 0.9, 0.3));
        kps.push(Keypoint::new(CocoKeypoint::LeftAnkle, 0.4, 0.8, 0.9, 0.3));
        // Invisible keypoint should not affect bbox
        kps.push(Keypoint::new(CocoKeypoint::RightAnkle, 0.9, 0.9, 0.1, 0.3));
        let person = PersonPose::new(kps, 0);
        let (x_min, y_min, x_max, y_max) = person.bounding_box;
        assert!((x_min - 0.2).abs() < 1e-5);
        assert!((y_min - 0.1).abs() < 1e-5);
        assert!((x_max - 0.4).abs() < 1e-5);
        assert!((y_max - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_person_pose_visible_keypoints() {
        let kps: Vec<Keypoint> = CocoKeypoint::all()
            .iter()
            .enumerate()
            .map(|(i, kp)| {
                // First 10 visible, last 7 not
                let conf = if i < 10 { 0.9 } else { 0.1 };
                Keypoint::new(*kp, 0.5, 0.5, conf, 0.3)
            })
            .collect();
        let person = PersonPose::new(kps, 0);
        assert_eq!(person.visible_keypoints().len(), 10);
    }

    #[test]
    fn test_person_pose_torso_width() {
        let mut kps: Vec<Keypoint> = Vec::new();
        kps.push(Keypoint::new(
            CocoKeypoint::LeftShoulder,
            0.2,
            0.3,
            0.9,
            0.3,
        ));
        kps.push(Keypoint::new(
            CocoKeypoint::RightShoulder,
            0.8,
            0.3,
            0.9,
            0.3,
        ));
        let person = PersonPose::new(kps, 0);
        let tw = person.torso_width().expect("should compute torso width");
        assert!(
            (tw - 0.6).abs() < 1e-5,
            "torso width should be 0.6, got {}",
            tw
        );
    }

    #[test]
    fn test_person_pose_is_upright() {
        // Nose above hip (y=0.1 < y=0.6)
        let mut kps: Vec<Keypoint> = Vec::new();
        kps.push(Keypoint::new(CocoKeypoint::Nose, 0.5, 0.1, 0.9, 0.3));
        kps.push(Keypoint::new(CocoKeypoint::LeftHip, 0.3, 0.6, 0.9, 0.3));
        let person = PersonPose::new(kps, 0);
        assert!(person.is_upright());

        // Inverted: nose below hip
        let mut kps2: Vec<Keypoint> = Vec::new();
        kps2.push(Keypoint::new(CocoKeypoint::Nose, 0.5, 0.9, 0.9, 0.3));
        kps2.push(Keypoint::new(CocoKeypoint::LeftHip, 0.3, 0.2, 0.9, 0.3));
        let person2 = PersonPose::new(kps2, 1);
        assert!(!person2.is_upright());
    }

    #[test]
    fn test_pose_result_filter_by_score() {
        let kps_high: Vec<Keypoint> = CocoKeypoint::all()
            .iter()
            .map(|kp| Keypoint::new(*kp, 0.5, 0.5, 0.9, 0.3))
            .collect();
        let kps_low: Vec<Keypoint> = CocoKeypoint::all()
            .iter()
            .map(|kp| Keypoint::new(*kp, 0.5, 0.5, 0.1, 0.3))
            .collect();
        let result = PoseEstimationResult {
            persons: vec![PersonPose::new(kps_high, 0), PersonPose::new(kps_low, 1)],
            width: 8,
            height: 8,
        };
        let filtered = result.filter_by_score(0.5);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].person_id, 0);
    }

    #[test]
    fn test_pose_pipeline_run() {
        let pipeline = PoseEstimationPipeline::new("vitpose-b");
        let image = make_image(8, 8);
        let result = pipeline.run(&image, 8, 8).expect("run should succeed");
        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
        // At least one person should be detected with a reasonable image
        assert!(result.num_persons() >= 1);
        // Each detected person should have 17 keypoints
        for p in &result.persons {
            assert_eq!(p.keypoints.len(), 17);
        }
    }

    #[test]
    fn test_pose_pipeline_with_boxes() {
        let pipeline = PoseEstimationPipeline::new("vitpose-b");
        let image = make_image(8, 8);
        let boxes = vec![(0.0_f32, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0)];
        let result = pipeline
            .run_with_boxes(&image, 8, 8, &boxes)
            .expect("run_with_boxes should succeed");
        // Two persons, one per box
        assert_eq!(result.num_persons(), 2);
        // Person IDs should correspond to box indices
        assert_eq!(result.persons[0].person_id, 0);
        assert_eq!(result.persons[1].person_id, 1);
    }

    #[test]
    fn test_pose_error_display() {
        let e1 = PoseEstimationError::InvalidImageDimensions;
        assert!(e1.to_string().contains("invalid image dimensions"));

        let e2 = PoseEstimationError::EmptyImage;
        assert!(e2.to_string().contains("empty"));
    }

    // -----------------------------------------------------------------------
    // Extended types and PosePostprocessor tests (15+)
    // -----------------------------------------------------------------------

    // 1. COCO skeleton has 17 keypoints
    #[test]
    fn test_coco_skeleton_keypoint_count() {
        let skel = PosePostprocessor::coco_skeleton();
        assert_eq!(skel.keypoints.len(), 17);
    }

    // 2. COCO skeleton has 19 connections
    #[test]
    fn test_coco_skeleton_connection_count() {
        let skel = PosePostprocessor::coco_skeleton();
        assert_eq!(skel.connections.len(), 19);
    }

    // 3. COCO skeleton first keypoint is "nose"
    #[test]
    fn test_coco_skeleton_first_keypoint_name() {
        let skel = PosePostprocessor::coco_skeleton();
        assert_eq!(skel.keypoints[0].name, "nose");
    }

    // 4. COCO skeleton last keypoint is "right_ankle"
    #[test]
    fn test_coco_skeleton_last_keypoint_name() {
        let skel = PosePostprocessor::coco_skeleton();
        assert_eq!(skel.keypoints[16].name, "right_ankle");
    }

    // 5. COCO skeleton default keypoints all have NotLabeled visibility
    #[test]
    fn test_coco_skeleton_default_visibility() {
        let skel = PosePostprocessor::coco_skeleton();
        for kp in &skel.keypoints {
            assert_eq!(kp.visibility, KeypointVisibility::NotLabeled);
        }
    }

    // 6. heatmap_to_keypoints: peak found correctly
    #[test]
    fn test_heatmap_to_keypoints_peak() {
        // 3x3 heatmap with peak at (row=1, col=2)
        let hm = vec![
            vec![0.0_f32, 0.1, 0.2],
            vec![0.0, 0.3, 0.9], // peak here
            vec![0.0, 0.1, 0.0],
        ];
        let result = PosePostprocessor::heatmap_to_keypoints(&[hm]);
        assert_eq!(result.len(), 1);
        let (x, y, conf) = result[0];
        // col=2 out of 3 cols: x = 2/2 = 1.0; row=1 out of 3 rows: y = 1/2 = 0.5
        assert!((x - 1.0).abs() < 1e-5, "x={x}");
        assert!((y - 0.5).abs() < 1e-5, "y={y}");
        assert!((conf - 0.9).abs() < 1e-5, "conf={conf}");
    }

    // 7. heatmap_to_keypoints: multiple heatmaps
    #[test]
    fn test_heatmap_to_keypoints_multiple() {
        let hm1 = vec![vec![1.0_f32, 0.0], vec![0.0, 0.0]];
        let hm2 = vec![vec![0.0_f32, 0.0], vec![0.0, 1.0]];
        let result = PosePostprocessor::heatmap_to_keypoints(&[hm1, hm2]);
        assert_eq!(result.len(), 2);
        let (x1, y1, _) = result[0];
        let (x2, y2, _) = result[1];
        // hm1 peak at col=0, row=0
        assert!((x1 - 0.0).abs() < 1e-5, "x1={x1}");
        assert!((y1 - 0.0).abs() < 1e-5, "y1={y1}");
        // hm2 peak at col=1, row=1
        assert!((x2 - 1.0).abs() < 1e-5, "x2={x2}");
        assert!((y2 - 1.0).abs() < 1e-5, "y2={y2}");
    }

    // 8. heatmap_to_keypoints: empty heatmap list
    #[test]
    fn test_heatmap_to_keypoints_empty() {
        let result = PosePostprocessor::heatmap_to_keypoints(&[]);
        assert!(result.is_empty());
    }

    // 9. normalize_keypoints: coordinates become [0,1]
    #[test]
    fn test_normalize_keypoints_basic() {
        let mut kps = vec![
            NamedKeypoint::new("a", 100.0, 200.0, 0.9, 0.3),
            NamedKeypoint::new("b", 50.0, 400.0, 0.8, 0.3),
        ];
        PosePostprocessor::normalize_keypoints(&mut kps, 200.0, 400.0);
        assert!((kps[0].x - 0.5).abs() < 1e-5, "kps[0].x={}", kps[0].x);
        assert!((kps[0].y - 0.5).abs() < 1e-5, "kps[0].y={}", kps[0].y);
        assert!((kps[1].x - 0.25).abs() < 1e-5, "kps[1].x={}", kps[1].x);
        assert!((kps[1].y - 1.0).abs() < 1e-5, "kps[1].y={}", kps[1].y);
    }

    // 10. normalize_keypoints: zero dimensions are no-ops
    #[test]
    fn test_normalize_keypoints_zero_dims() {
        let mut kps = vec![NamedKeypoint::new("a", 50.0, 50.0, 0.9, 0.3)];
        PosePostprocessor::normalize_keypoints(&mut kps, 0.0, 0.0);
        // Should not panic and values remain unchanged
        assert!((kps[0].x - 50.0).abs() < 1e-5);
    }

    // 11. normalize_keypoints: clamping to [0,1]
    #[test]
    fn test_normalize_keypoints_clamped() {
        let mut kps = vec![NamedKeypoint::new("a", 300.0, -10.0, 0.9, 0.3)];
        PosePostprocessor::normalize_keypoints(&mut kps, 200.0, 200.0);
        assert!(kps[0].x <= 1.0, "x should be clamped to 1.0");
        assert!(kps[0].y >= 0.0, "y should be clamped to 0.0");
    }

    // 12. oks_score: identical keypoints give score ~1.0
    #[test]
    fn test_oks_score_identical() {
        let kps: Vec<NamedKeypoint> = (0..3)
            .map(|i| NamedKeypoint::new(&format!("kp{i}"), 0.1 * i as f32, 0.1, 0.9, 0.3))
            .collect();
        let sigmas = vec![0.1_f32; 3];
        let score = PosePostprocessor::oks_score(&kps, &kps, &sigmas);
        assert!(
            score > 0.95,
            "identical keypoints should have OKS ~1.0, got {score}"
        );
    }

    // 13. oks_score: far-apart keypoints give low score
    #[test]
    fn test_oks_score_far_apart() {
        let pred: Vec<NamedKeypoint> = vec![NamedKeypoint::new("a", 0.0, 0.0, 0.9, 0.3)];
        let gt: Vec<NamedKeypoint> = vec![NamedKeypoint::new("a", 1.0, 1.0, 0.9, 0.3)];
        let sigmas = vec![0.05_f32];
        let score = PosePostprocessor::oks_score(&pred, &gt, &sigmas);
        assert!(
            score < 0.1,
            "far-apart keypoints should have low OKS, got {score}"
        );
    }

    // 14. pck_accuracy: perfect prediction gives 1.0
    #[test]
    fn test_pck_accuracy_perfect() {
        let kps: Vec<NamedKeypoint> = (0..5)
            .map(|i| NamedKeypoint::new(&format!("kp{i}"), 0.1 * i as f32, 0.2, 0.9, 0.3))
            .collect();
        let acc = PosePostprocessor::pck_accuracy(&kps, &kps, 0.05);
        assert!(
            (acc - 1.0).abs() < 1e-5,
            "perfect match should give PCK 1.0, got {acc}"
        );
    }

    // 15. pck_accuracy: all predictions off by more than threshold gives 0.0
    #[test]
    fn test_pck_accuracy_zero() {
        let pred: Vec<NamedKeypoint> = vec![NamedKeypoint::new("a", 0.0, 0.0, 0.9, 0.3)];
        let gt: Vec<NamedKeypoint> = vec![NamedKeypoint::new("a", 0.9, 0.9, 0.9, 0.3)];
        let acc = PosePostprocessor::pck_accuracy(&pred, &gt, 0.05);
        assert!(
            (acc).abs() < 1e-5,
            "all-off predictions should give PCK 0.0, got {acc}"
        );
    }

    // 16. KeypointVisibility states
    #[test]
    fn test_keypoint_visibility_states() {
        let visible = NamedKeypoint::new("nose", 0.5, 0.5, 0.9, 0.3);
        assert_eq!(visible.visibility, KeypointVisibility::Visible);

        let occluded = NamedKeypoint::new("ear", 0.5, 0.5, 0.1, 0.3);
        assert_eq!(occluded.visibility, KeypointVisibility::Occluded);

        let not_labeled = NamedKeypoint::new("hip", 0.5, 0.5, 0.0, 0.3);
        assert_eq!(not_labeled.visibility, KeypointVisibility::NotLabeled);
    }

    // 17. PoseResult construction
    #[test]
    fn test_pose_result_construction() {
        let skel = PosePostprocessor::coco_skeleton();
        let result = PoseResult {
            skeleton: skel,
            bbox: (0.1, 0.1, 0.9, 0.9),
            score: 0.85,
        };
        assert_eq!(result.skeleton.keypoints.len(), 17);
        assert!((result.score - 0.85).abs() < 1e-5);
    }

    // 18. OKS: empty inputs return 0.0
    #[test]
    fn test_oks_empty_inputs() {
        let kps: Vec<NamedKeypoint> = vec![];
        assert_eq!(PosePostprocessor::oks_score(&kps, &kps, &[0.1]), 0.0);
    }

    // 19. PCK: empty inputs return 0.0
    #[test]
    fn test_pck_empty_inputs() {
        let kps: Vec<NamedKeypoint> = vec![];
        assert_eq!(PosePostprocessor::pck_accuracy(&kps, &kps, 0.1), 0.0);
    }

    // 20. Skeleton connections indices stay in bounds
    #[test]
    fn test_coco_skeleton_connection_bounds() {
        let skel = PosePostprocessor::coco_skeleton();
        let n = skel.keypoints.len();
        for &(a, b) in &skel.connections {
            assert!(a < n, "connection start {a} out of range [0,{n})");
            assert!(b < n, "connection end {b} out of range [0,{n})");
        }
    }
}
