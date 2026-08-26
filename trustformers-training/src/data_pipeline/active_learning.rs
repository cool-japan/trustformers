//! Active learning: query strategies, uncertainty/disagreement measures, sampling, annotation and quality control.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};

use super::pipeline::DataSample;

/// Active learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveLearningConfig {
    /// Query strategy
    pub query_strategy: QueryStrategy,
    /// Sampling configuration
    pub sampling: SamplingConfig,
    /// Annotation configuration
    pub annotation: AnnotationConfig,
    /// Integration settings
    pub integration: ActiveLearningIntegration,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveLearningIntegration {
    /// Update frequency
    pub update_frequency: usize,
    /// Minimum new samples before update
    pub min_new_samples: usize,
    /// Retrain from scratch
    pub retrain_from_scratch: bool,
}
pub struct ActiveLearningManager {
    pub config: ActiveLearningConfig,
    pub query_pool: Vec<DataSample>,
    pub stats: ActiveLearningStats,
}
impl ActiveLearningManager {
    pub fn new() -> Self {
        Self {
            config: ActiveLearningConfig {
                query_strategy: QueryStrategy::UncertaintySampling {
                    uncertainty_measure: UncertaintyMeasure::Entropy,
                },
                sampling: SamplingConfig {
                    batch_size: 10,
                    budget: 1000,
                    diversity_constraint: None,
                },
                annotation: AnnotationConfig {
                    source: AnnotationSource::Human {
                        annotator_pool: vec![],
                    },
                    quality_control: QualityControl {
                        multi_annotation: false,
                        agreement_threshold: 0.8,
                        assessment_method: QualityAssessmentMethod::InterAnnotatorAgreement,
                    },
                },
                integration: ActiveLearningIntegration {
                    update_frequency: 100,
                    min_new_samples: 10,
                    retrain_from_scratch: false,
                },
            },
            query_pool: vec![],
            stats: ActiveLearningStats {
                queries_made: 0,
                annotations_received: 0,
                model_improvement: 0.0,
            },
        }
    }
}
impl Default for ActiveLearningManager {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Debug, Clone)]
pub struct ActiveLearningStats {
    pub queries_made: usize,
    pub annotations_received: usize,
    pub model_improvement: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationConfig {
    /// Annotation source
    pub source: AnnotationSource,
    /// Quality control
    pub quality_control: QualityControl,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnnotationSource {
    /// Human annotators
    Human { annotator_pool: Vec<String> },
    /// Automatic annotation
    Automatic {
        model_path: String,
        confidence_threshold: f64,
    },
    /// Hybrid human + automatic
    Hybrid {
        automatic_threshold: f64,
        human_verification: bool,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoreSetMethod {
    /// K-center greedy
    KCenterGreedy,
    /// K-means++
    KMeansPlusPlus,
    /// Facility location
    FacilityLocation,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DisagreementMeasure {
    /// Vote entropy
    VoteEntropy,
    /// KL divergence
    KLDivergence,
    /// Average KL divergence
    AverageKLDivergence,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityConstraint {
    /// Diversity measure
    pub measure: DiversityMeasure,
    /// Minimum diversity threshold
    pub threshold: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiversityMeasure {
    /// Cosine similarity
    CosineSimilarity,
    /// Euclidean distance
    EuclideanDistance,
    /// Jaccard similarity
    JaccardSimilarity,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityAssessmentMethod {
    /// Inter-annotator agreement
    InterAnnotatorAgreement,
    /// Gold standard comparison
    GoldStandard { gold_set_path: String },
    /// Model-based quality assessment
    ModelBased { quality_model_path: String },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityControl {
    /// Multiple annotations per sample
    pub multi_annotation: bool,
    /// Agreement threshold
    pub agreement_threshold: f64,
    /// Quality assessment method
    pub assessment_method: QualityAssessmentMethod,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryStrategy {
    /// Uncertainty sampling
    UncertaintySampling {
        uncertainty_measure: UncertaintyMeasure,
    },
    /// Query by committee
    QueryByCommittee {
        committee_size: usize,
        disagreement_measure: DisagreementMeasure,
    },
    /// Expected gradient length
    ExpectedGradientLength,
    /// Bayesian active learning by disagreement
    BALD,
    /// Core-set selection
    CoreSet { selection_method: CoreSetMethod },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Batch size for active learning queries
    pub batch_size: usize,
    /// Sampling budget
    pub budget: usize,
    /// Diversity constraint
    pub diversity_constraint: Option<DiversityConstraint>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UncertaintyMeasure {
    /// Least confidence
    LeastConfidence,
    /// Margin sampling
    MarginSampling,
    /// Entropy
    Entropy,
    /// Variation ratios
    VariationRatios,
}
