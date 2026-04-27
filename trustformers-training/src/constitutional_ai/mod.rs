//! Constitutional AI (CAI) training infrastructure.
//!
//! Implements the critique-revision self-improvement loop from
//! "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022).
//!
//! The core idea: a set of principles ("the constitution") guides the model to
//! critique its own potentially harmful outputs and then revise them to be more
//! aligned with the stated principles.  The resulting (initial, revised) pairs
//! are turned into preference data for subsequent reward model or RLHF training.

// ──────────────────────────────────────────────
// ConstitutionalPrinciple
// ──────────────────────────────────────────────

/// A single principle from the constitution.
///
/// Each principle provides a *critique request* (ask the model to identify
/// violations) and a *revision request* (ask the model to fix them).
#[derive(Debug, Clone)]
pub struct CaiConstitutionalPrinciple {
    /// Unique identifier for this principle (e.g. `"no_harm"`).
    pub id: String,
    /// The critique prompt appended after the response.
    pub critique_request: String,
    /// The revision prompt used to improve the response.
    pub revision_request: String,
    /// Short human-readable name (e.g. `"HHH"`, `"no_discrimination"`).
    pub name: String,
}

// ──────────────────────────────────────────────
// Constitution
// ──────────────────────────────────────────────

/// A collection of [`CaiConstitutionalPrinciple`]s.
#[derive(Debug, Clone)]
pub struct Constitution {
    /// Ordered list of principles.
    pub principles: Vec<CaiConstitutionalPrinciple>,
}

impl Constitution {
    /// Create a constitution from a list of principles.
    pub fn new(principles: Vec<CaiConstitutionalPrinciple>) -> Self {
        Self { principles }
    }

    /// Return a standard **Helpful, Harmless and Honest (HHH)** constitution
    /// with five core principles.
    pub fn hhh_constitution() -> Self {
        let principles = vec![
            CaiConstitutionalPrinciple {
                id: "no_harmful_content".to_string(),
                name: "no_harmful_content".to_string(),
                critique_request: "Identify specific ways in which the assistant's last \
                    response is harmful, dangerous, or could cause physical or psychological \
                    harm to people."
                    .to_string(),
                revision_request: "Please rewrite the assistant response to remove any \
                    harmful, dangerous, or potentially injurious content, making it safe \
                    for all audiences."
                    .to_string(),
            },
            CaiConstitutionalPrinciple {
                id: "no_dishonesty".to_string(),
                name: "no_dishonesty".to_string(),
                critique_request: "Identify specific ways in which the assistant's last \
                    response contains falsehoods, misleading information, or dishonest claims."
                    .to_string(),
                revision_request: "Please rewrite the assistant response to ensure all \
                    statements are accurate, honest, and not misleading."
                    .to_string(),
            },
            CaiConstitutionalPrinciple {
                id: "no_manipulation".to_string(),
                name: "no_manipulation".to_string(),
                critique_request: "Identify specific ways in which the assistant's last \
                    response uses manipulative, deceptive, or psychologically coercive \
                    techniques."
                    .to_string(),
                revision_request: "Please rewrite the assistant response to remove any \
                    manipulative or deceptive rhetoric, ensuring a straightforward and \
                    respectful interaction."
                    .to_string(),
            },
            CaiConstitutionalPrinciple {
                id: "privacy_protection".to_string(),
                name: "privacy_protection".to_string(),
                critique_request: "Identify specific ways in which the assistant's last \
                    response violates individual privacy, shares personally identifiable \
                    information, or encourages surveillance."
                    .to_string(),
                revision_request: "Please rewrite the assistant response to respect privacy, \
                    remove any PII, and refrain from encouraging privacy-violating behaviour."
                    .to_string(),
            },
            CaiConstitutionalPrinciple {
                id: "no_discrimination".to_string(),
                name: "no_discrimination".to_string(),
                critique_request: "Identify specific ways in which the assistant's last \
                    response contains biased, discriminatory, or prejudiced content related \
                    to race, gender, religion, nationality, sexuality, or other protected \
                    characteristics."
                    .to_string(),
                revision_request: "Please rewrite the assistant response to remove any \
                    biased, discriminatory, or prejudiced language and treat all groups \
                    fairly and respectfully."
                    .to_string(),
            },
        ];
        Self { principles }
    }

    /// Number of principles in the constitution.
    pub fn len(&self) -> usize {
        self.principles.len()
    }

    /// Returns `true` when the constitution has no principles.
    pub fn is_empty(&self) -> bool {
        self.principles.is_empty()
    }

    /// Look up a principle by its `id` field.
    pub fn get_principle(&self, id: &str) -> Option<&CaiConstitutionalPrinciple> {
        self.principles.iter().find(|p| p.id == id)
    }

    /// Deterministically select a principle by index (wraps around).
    ///
    /// Returns `principles[idx % len()]`.  Panics only if the constitution is
    /// empty — callers should ensure the constitution has at least one principle.
    pub fn random_principle_by_index(&self, idx: usize) -> &CaiConstitutionalPrinciple {
        let n = self.principles.len();
        debug_assert!(n > 0, "Constitution must have at least one principle");
        &self.principles[idx % n]
    }
}

// ──────────────────────────────────────────────
// CaiConfig
// ──────────────────────────────────────────────

/// Configuration for a Constitutional AI training run.
#[derive(Debug, Clone)]
pub struct CaiConfig {
    /// Number of critique-revision cycles applied per sample (default 1).
    pub num_critique_revisions: usize,
    /// Number of principles applied to each example (default 1).
    pub num_principles_per_sample: usize,
    /// Maximum token budget for the generated critique.
    pub max_critique_length: usize,
    /// Maximum token budget for the revised response.
    pub max_revision_length: usize,
    /// Template for the critique prompt.
    ///
    /// Must contain the placeholders `{critique_request}` and `{response}`.
    pub feedback_model_prompt_template: String,
    /// Template for the revision prompt.
    ///
    /// Must contain the placeholders `{revision_request}`, `{response}`, and
    /// `{critique}`.
    pub revision_prompt_template: String,
    /// Whether the model critiques its own outputs (true) or a separate model
    /// is used (false).
    pub use_self_critique: bool,
}

impl Default for CaiConfig {
    fn default() -> Self {
        Self {
            num_critique_revisions: 1,
            num_principles_per_sample: 1,
            max_critique_length: 256,
            max_revision_length: 512,
            feedback_model_prompt_template:
                "Response: {response}\n\nCritique request: {critique_request}\n\nCritique:"
                    .to_string(),
            revision_prompt_template:
                "Response: {response}\n\nCritique: {critique}\n\nRevision request: \
                 {revision_request}\n\nRevised response:"
                    .to_string(),
            use_self_critique: true,
        }
    }
}

// ──────────────────────────────────────────────
// CaiExample
// ──────────────────────────────────────────────

/// One Constitutional AI example capturing the full critique-revision cycle.
#[derive(Debug, Clone)]
pub struct CaiExample {
    /// The original user prompt.
    pub prompt: String,
    /// The model's potentially problematic initial response.
    pub initial_response: String,
    /// The model's self-critique (populated after critique step).
    pub critique: Option<String>,
    /// The improved response after applying the revision request.
    pub revised_response: Option<String>,
    /// Short name of the principle that was applied.
    pub principle_used: String,
    /// Principle id that was applied.
    pub applied_principle_id: String,
}

// ──────────────────────────────────────────────
// CaiDataPoint
// ──────────────────────────────────────────────

/// A preference data point derived from a CAI revision cycle, suitable for
/// reward model or DPO training.
#[derive(Debug, Clone)]
pub struct CaiDataPoint {
    /// The original user prompt.
    pub prompt: String,
    /// The revised (preferred) response.
    pub chosen: String,
    /// The initial (less preferred) response.
    pub rejected: String,
    /// The principle text that explains why chosen is preferred.
    pub principle: String,
}

// ──────────────────────────────────────────────
// CaiPipeline
// ──────────────────────────────────────────────

/// Orchestrates the Constitutional AI critique-revision pipeline.
pub struct CaiPipeline {
    /// The constitution guiding critique and revision.
    pub constitution: Constitution,
    /// Pipeline configuration.
    pub config: CaiConfig,
}

impl CaiPipeline {
    /// Construct a new pipeline.
    pub fn new(constitution: Constitution, config: CaiConfig) -> Self {
        Self { constitution, config }
    }

    /// Build the critique prompt by filling the template placeholders.
    pub fn format_critique_prompt(
        &self,
        response: &str,
        principle: &CaiConstitutionalPrinciple,
    ) -> String {
        self.config
            .feedback_model_prompt_template
            .replace("{response}", response)
            .replace("{critique_request}", &principle.critique_request)
    }

    /// Build the revision prompt by filling the template placeholders.
    pub fn format_revision_prompt(
        &self,
        response: &str,
        critique: &str,
        principle: &CaiConstitutionalPrinciple,
    ) -> String {
        self.config
            .revision_prompt_template
            .replace("{response}", response)
            .replace("{critique}", critique)
            .replace("{revision_request}", &principle.revision_request)
    }

    /// Convert a [`CaiExample`] into a [`CaiDataPoint`] for training.
    ///
    /// Returns `None` when:
    /// - `revised_response` is `None`, or
    /// - the revised response is identical to the initial response (no improvement).
    pub fn create_training_data(example: CaiExample) -> Option<CaiDataPoint> {
        let revised = example.revised_response?;
        if revised == example.initial_response {
            return None;
        }
        Some(CaiDataPoint {
            prompt: example.prompt,
            chosen: revised,
            rejected: example.initial_response,
            principle: example.principle_used,
        })
    }

    /// Select `num_principles_per_sample` principles for a given example index.
    ///
    /// Principles are chosen deterministically using
    /// `random_principle_by_index(example_idx + i)` for `i` in
    /// `0..num_principles_per_sample`.
    pub fn select_principles_for_example(
        &self,
        example_idx: usize,
    ) -> Vec<&CaiConstitutionalPrinciple> {
        (0..self.config.num_principles_per_sample)
            .map(|i| self.constitution.random_principle_by_index(example_idx + i))
            .collect()
    }

    /// Heuristic harmlessness-improvement score in `[0, 1]`.
    ///
    /// Counts occurrences of a small list of harmful keywords in both the
    /// initial and revised response and returns the relative reduction.
    ///
    /// `score = (initial_matches − revised_matches) / max(initial_matches, 1)`
    pub fn evaluate_harmlessness_improvement(initial: &str, revised: &str) -> f32 {
        const HARMFUL_KEYWORDS: &[&str] =
            &["harm", "kill", "hurt", "dangerous", "illegal", "attack"];

        let count = |text: &str| -> usize {
            let lower = text.to_lowercase();
            HARMFUL_KEYWORDS.iter().filter(|&&kw| lower.contains(kw)).count()
        };

        let initial_matches = count(initial) as f32;
        let revised_matches = count(revised) as f32;

        (initial_matches - revised_matches) / initial_matches.max(1.0)
    }
}

// ──────────────────────────────────────────────
// CaiTrainingDataset
// ──────────────────────────────────────────────

/// A collection of [`CaiDataPoint`]s with management utilities.
#[derive(Debug, Default)]
pub struct CaiTrainingDataset {
    /// The stored data points.
    pub data_points: Vec<CaiDataPoint>,
}

impl CaiTrainingDataset {
    /// Create an empty dataset.
    pub fn new() -> Self {
        Self { data_points: Vec::new() }
    }

    /// Append a data point.
    pub fn add(&mut self, point: CaiDataPoint) {
        self.data_points.push(point);
    }

    /// Number of data points.
    pub fn len(&self) -> usize {
        self.data_points.len()
    }

    /// Returns `true` when the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.data_points.is_empty()
    }

    /// Return all data points whose principle text contains `principle_id`.
    ///
    /// Matches against the `principle` field (which stores the principle name /
    /// critique text), checking for an exact `principle_id` substring.
    pub fn filter_by_principle(&self, principle_id: &str) -> Vec<&CaiDataPoint> {
        self.data_points
            .iter()
            .filter(|dp| dp.principle.contains(principle_id))
            .collect()
    }

    /// Return preference pairs as `(prompt, chosen, rejected)` tuples.
    pub fn as_preference_pairs(&self) -> Vec<(String, String, String)> {
        self.data_points
            .iter()
            .map(|dp| (dp.prompt.clone(), dp.chosen.clone(), dp.rejected.clone()))
            .collect()
    }
}

// ──────────────────────────────────────────────
// ConstitutionPrinciple (new rich wrapper)
// ──────────────────────────────────────────────

/// Category of a constitutional principle, used for filtering and reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrincipleCategory {
    Harmlessness,
    Helpfulness,
    Honesty,
    Ethics,
}

/// A richer constitutional principle with an explicit category.
#[derive(Debug, Clone)]
pub struct ConstitutionPrinciple {
    pub name: String,
    pub critique_prompt: String,
    pub revision_prompt: String,
    pub category: PrincipleCategory,
}

/// Return the default Anthropic Constitutional AI principles (16 principles).
///
/// These are the publicly documented CAI principles covering harmlessness,
/// honesty, helpfulness, and ethical behaviour.
pub fn anthropic_constitution() -> Vec<ConstitutionPrinciple> {
    vec![
        ConstitutionPrinciple {
            name: "no_harmful_content".to_string(),
            critique_prompt: "Identify specific ways in which the assistant's response is \
                harmful, unethical, racist, sexist, toxic, dangerous, or illegal."
                .to_string(),
            revision_prompt: "Please rewrite the assistant response to remove any harmful, \
                unethical, racist, sexist, toxic, dangerous, or illegal content."
                .to_string(),
            category: PrincipleCategory::Harmlessness,
        },
        ConstitutionPrinciple {
            name: "no_dangerous_content".to_string(),
            critique_prompt: "Identify content that could be used to cause serious physical harm \
                or death to people, including instructions for weapons, drugs, or violence."
                .to_string(),
            revision_prompt: "Rewrite the response to remove content that could be used to cause \
                serious harm to people."
                .to_string(),
            category: PrincipleCategory::Harmlessness,
        },
        ConstitutionPrinciple {
            name: "no_discriminatory_content".to_string(),
            critique_prompt: "Identify ways in which the assistant's last response is \
                discriminatory, biased, or prejudiced against groups of people."
                .to_string(),
            revision_prompt: "Rewrite the assistant response to remove discriminatory language \
                and treat all groups fairly."
                .to_string(),
            category: PrincipleCategory::Harmlessness,
        },
        ConstitutionPrinciple {
            name: "no_privacy_violation".to_string(),
            critique_prompt: "Identify ways in which the assistant's response violates privacy, \
                shares personal information, or encourages surveillance."
                .to_string(),
            revision_prompt: "Rewrite the response to respect individual privacy and remove any \
                personally identifiable information."
                .to_string(),
            category: PrincipleCategory::Harmlessness,
        },
        ConstitutionPrinciple {
            name: "no_deception".to_string(),
            critique_prompt: "Identify ways in which the assistant's response is deceptive, \
                misleading, or contains falsehoods."
                .to_string(),
            revision_prompt: "Rewrite the response to ensure all statements are accurate, \
                honest, and non-deceptive."
                .to_string(),
            category: PrincipleCategory::Honesty,
        },
        ConstitutionPrinciple {
            name: "no_manipulation".to_string(),
            critique_prompt: "Identify ways in which the assistant's response uses manipulative, \
                coercive, or psychologically harmful techniques."
                .to_string(),
            revision_prompt: "Rewrite the response to remove manipulative or coercive \
                rhetoric and ensure it is respectful and honest."
                .to_string(),
            category: PrincipleCategory::Honesty,
        },
        ConstitutionPrinciple {
            name: "factual_accuracy".to_string(),
            critique_prompt: "Identify factual errors, unsupported claims, or misleading \
                statistics in the assistant's response."
                .to_string(),
            revision_prompt: "Rewrite the response to correct all factual errors and clearly \
                distinguish between facts and opinions."
                .to_string(),
            category: PrincipleCategory::Honesty,
        },
        ConstitutionPrinciple {
            name: "acknowledge_uncertainty".to_string(),
            critique_prompt: "Identify places where the assistant expressed false certainty or \
                failed to acknowledge genuine uncertainty."
                .to_string(),
            revision_prompt: "Rewrite the response to appropriately convey uncertainty and \
                avoid overconfident claims."
                .to_string(),
            category: PrincipleCategory::Honesty,
        },
        ConstitutionPrinciple {
            name: "be_helpful".to_string(),
            critique_prompt: "Identify ways in which the assistant's response failed to \
                address the user's actual need or was unnecessarily unhelpful."
                .to_string(),
            revision_prompt: "Rewrite the response to be more genuinely helpful, clearly \
                addressing the user's question or need."
                .to_string(),
            category: PrincipleCategory::Helpfulness,
        },
        ConstitutionPrinciple {
            name: "clear_communication".to_string(),
            critique_prompt: "Identify ways in which the assistant's response was unclear, \
                confusing, or poorly structured."
                .to_string(),
            revision_prompt: "Rewrite the response to be clear, well-structured, and easy \
                to understand."
                .to_string(),
            category: PrincipleCategory::Helpfulness,
        },
        ConstitutionPrinciple {
            name: "avoid_refusal_when_safe".to_string(),
            critique_prompt: "Identify whether the assistant unnecessarily refused to help \
                with a safe and reasonable request."
                .to_string(),
            revision_prompt: "Rewrite the response to helpfully assist with the request if \
                it is safe and reasonable to do so."
                .to_string(),
            category: PrincipleCategory::Helpfulness,
        },
        ConstitutionPrinciple {
            name: "no_copyright_violation".to_string(),
            critique_prompt: "Identify ways in which the assistant's response reproduces \
                copyrighted material without appropriate transformation or attribution."
                .to_string(),
            revision_prompt: "Rewrite the response to avoid reproducing copyrighted material \
                and instead paraphrase or summarise in original language."
                .to_string(),
            category: PrincipleCategory::Ethics,
        },
        ConstitutionPrinciple {
            name: "no_hate_speech".to_string(),
            critique_prompt: "Identify whether the assistant's response contains hate speech, \
                slurs, or content that degrades or dehumanises any group of people."
                .to_string(),
            revision_prompt: "Rewrite the response to remove all hate speech and ensure it \
                treats every group of people with dignity and respect."
                .to_string(),
            category: PrincipleCategory::Ethics,
        },
        ConstitutionPrinciple {
            name: "respect_autonomy".to_string(),
            critique_prompt: "Identify ways in which the assistant's response fails to \
                respect the user's autonomy or unnecessarily paternalistic."
                .to_string(),
            revision_prompt: "Rewrite the response to respect the user's autonomy while \
                providing relevant information."
                .to_string(),
            category: PrincipleCategory::Ethics,
        },
        ConstitutionPrinciple {
            name: "no_illegal_advice".to_string(),
            critique_prompt: "Identify whether the assistant provides advice that encourages \
                or facilitates clearly illegal activities."
                .to_string(),
            revision_prompt: "Rewrite the response to not encourage or facilitate illegal \
                activities."
                .to_string(),
            category: PrincipleCategory::Ethics,
        },
        ConstitutionPrinciple {
            name: "protect_minors".to_string(),
            critique_prompt: "Identify content in the assistant's response that could be \
                harmful to minors or that is age-inappropriate."
                .to_string(),
            revision_prompt: "Rewrite the response to be appropriate for all audiences \
                and to protect the wellbeing of minors."
                .to_string(),
            category: PrincipleCategory::Harmlessness,
        },
    ]
}

// ──────────────────────────────────────────────
// ConstitutionalAiTrainer
// ──────────────────────────────────────────────

/// Configuration for a full Constitutional AI training run.
#[derive(Debug, Clone)]
pub struct CaiTrainingConfig {
    /// Number of critique-revision cycles per sample (default 1).
    pub num_critique_revisions: usize,
    /// Sampling temperature for critique generation (stub — no real model).
    pub critique_temperature: f32,
    /// Sampling temperature for revision generation (stub — no real model).
    pub revision_temperature: f32,
    /// Identifier of the feedback model to use.
    pub feedback_model: String,
}

impl Default for CaiTrainingConfig {
    fn default() -> Self {
        Self {
            num_critique_revisions: 1,
            critique_temperature: 0.7,
            revision_temperature: 0.5,
            feedback_model: "claude-3-sonnet".to_string(),
        }
    }
}

/// Statistics from a red-teaming run.
#[derive(Debug, Clone, Default)]
pub struct RedTeamStats {
    pub total_prompts: u64,
    pub harmful_detected: u64,
    pub revised_successfully: u64,
    /// Average improvement in harmlessness score across revised samples.
    pub revision_score_improvement: f32,
}

/// Full Constitutional AI training pipeline combining a constitution, config,
/// and accumulated red-team statistics.
pub struct ConstitutionalAiTrainer {
    pub constitution: Vec<ConstitutionPrinciple>,
    pub config: CaiTrainingConfig,
    pub red_team_stats: RedTeamStats,
}

impl ConstitutionalAiTrainer {
    /// Create a new trainer with the given constitution and config.
    pub fn new(constitution: Vec<ConstitutionPrinciple>, config: CaiTrainingConfig) -> Self {
        Self {
            constitution,
            config,
            red_team_stats: RedTeamStats::default(),
        }
    }

    /// Create a trainer pre-loaded with the default Anthropic constitution.
    pub fn with_anthropic_constitution() -> Self {
        Self::new(anthropic_constitution(), CaiTrainingConfig::default())
    }

    /// Return all principles belonging to the given category.
    pub fn principles_by_category(&self, category: &PrincipleCategory) -> Vec<&ConstitutionPrinciple> {
        self.constitution
            .iter()
            .filter(|p| &p.category == category)
            .collect()
    }

    /// Record the outcome of processing a single red-team prompt.
    ///
    /// `initial_score` and `revised_score` are harmlessness scores in `[0, 1]`
    /// (higher is safer).
    pub fn record_red_team_result(
        &mut self,
        was_harmful: bool,
        revised_successfully: bool,
        score_improvement: f32,
    ) {
        self.red_team_stats.total_prompts += 1;
        if was_harmful {
            self.red_team_stats.harmful_detected += 1;
        }
        if revised_successfully {
            self.red_team_stats.revised_successfully += 1;
        }
        // Running average update.
        let n = self.red_team_stats.total_prompts as f32;
        self.red_team_stats.revision_score_improvement +=
            (score_improvement - self.red_team_stats.revision_score_improvement) / n;
    }
}

// ──────────────────────────────────────────────
// HarmScorer
// ──────────────────────────────────────────────

/// A rule-based harm scorer that flags text containing known toxic keywords.
///
/// Scores range from `0.0` (fully safe) to `1.0` (maximally harmful).
pub struct HarmScorer {
    pub toxic_keywords: Vec<String>,
    pub threshold: f32,
}

impl HarmScorer {
    /// Create a new scorer with an explicit keyword list and decision threshold.
    pub fn new(toxic_keywords: Vec<String>, threshold: f32) -> Self {
        Self { toxic_keywords, threshold }
    }

    /// Create a scorer with the built-in default keyword list.
    pub fn with_defaults() -> Self {
        let keywords = vec![
            "harm", "kill", "murder", "attack", "weapon", "explosive", "poison",
            "dangerous", "illegal", "threat", "bomb", "abuse", "hate", "violence",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        Self::new(keywords, 0.3)
    }

    /// Score text on `[0.0, 1.0]` based on keyword density.
    ///
    /// `score = min(matches / total_keywords, 1.0)`
    pub fn score(&self, text: &str) -> f32 {
        if self.toxic_keywords.is_empty() {
            return 0.0;
        }
        let lower = text.to_lowercase();
        let matches = self
            .toxic_keywords
            .iter()
            .filter(|kw| lower.contains(kw.as_str()))
            .count();
        (matches as f32 / self.toxic_keywords.len() as f32).min(1.0)
    }

    /// Return `true` when `score(text) >= threshold`.
    pub fn is_harmful(&self, text: &str) -> bool {
        self.score(text) >= self.threshold
    }

    /// Return the list of keywords that were triggered by the given text.
    pub fn explain(&self, text: &str) -> Vec<String> {
        let lower = text.to_lowercase();
        self.toxic_keywords
            .iter()
            .filter(|kw| lower.contains(kw.as_str()))
            .cloned()
            .collect()
    }
}

// ──────────────────────────────────────────────
// RLAIF Data Generator
// ──────────────────────────────────────────────

/// A scored preference pair produced by the RLAIF data generator.
#[derive(Debug, Clone)]
pub struct PreferencePair {
    pub chosen: String,
    pub rejected: String,
    pub chosen_score: f32,
    pub rejected_score: f32,
}

/// Preference model used by [`RlaifDataGenerator`] to rank candidate responses.
#[derive(Debug, Clone)]
pub enum PreferenceModel {
    /// Always maximises harmlessness (lower harm score wins).
    ConstantHarmlessness,
    /// Prefers longer responses (as a proxy for helpfulness).
    LengthBased,
    /// Weighted combination of harmlessness and helpfulness.
    Combined { harmless_weight: f32, helpful_weight: f32 },
}

/// Generates RLAIF preference pairs from a set of candidate responses to a
/// single prompt.
pub struct RlaifDataGenerator {
    pub num_candidates: usize,
    pub preference_model: PreferenceModel,
}

impl RlaifDataGenerator {
    /// Create a new generator.
    pub fn new(num_candidates: usize, preference_model: PreferenceModel) -> Self {
        Self { num_candidates, preference_model }
    }

    /// Score a single response according to the preference model.
    ///
    /// Higher scores are better.
    pub fn score(&self, response: &str) -> f32 {
        match &self.preference_model {
            PreferenceModel::ConstantHarmlessness => {
                let scorer = HarmScorer::with_defaults();
                // Invert: lower harm → higher score
                1.0 - scorer.score(response)
            }
            PreferenceModel::LengthBased => {
                // Normalise length to [0, 1] with a soft cap at 1000 chars.
                let len = response.len().min(1000) as f32;
                len / 1000.0
            }
            PreferenceModel::Combined { harmless_weight, helpful_weight } => {
                let scorer = HarmScorer::with_defaults();
                let harmless_score = 1.0 - scorer.score(response);
                let helpful_score = response.len().min(1000) as f32 / 1000.0;
                let total = harmless_weight + helpful_weight;
                if total <= 0.0 {
                    return 0.5;
                }
                (harmless_weight * harmless_score + helpful_weight * helpful_score) / total
            }
        }
    }

    /// Given a prompt and N candidate responses, generate all non-trivial
    /// preference pairs (chosen score strictly > rejected score).
    ///
    /// Each candidate is scored; the highest-scoring response is paired with
    /// every lower-scoring one.
    pub fn generate_preferences(&self, _prompt: &str, candidates: &[String]) -> Vec<PreferencePair> {
        if candidates.len() < 2 {
            return Vec::new();
        }

        // Score all candidates.
        let scored: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.score(c)))
            .collect();

        // Find the best candidate.
        let best_idx = scored
            .iter()
            .enumerate()
            .max_by(|(_, (_, sa)), (_, (_, sb))| {
                sa.partial_cmp(sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, (i, _))| *i)
            .unwrap_or(0);

        let chosen_score = scored[best_idx].1;
        let chosen_text = candidates[best_idx].clone();

        scored
            .iter()
            .filter(|(i, score)| *i != best_idx && *score < chosen_score)
            .map(|(i, rejected_score)| PreferencePair {
                chosen: chosen_text.clone(),
                rejected: candidates[*i].clone(),
                chosen_score,
                rejected_score: *rejected_score,
            })
            .collect()
    }
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_principle(id: &str, name: &str) -> CaiConstitutionalPrinciple {
        CaiConstitutionalPrinciple {
            id: id.to_string(),
            name: name.to_string(),
            critique_request: format!("Critique for {}", name),
            revision_request: format!("Revise for {}", name),
        }
    }

    // ── Test 1: constitution creation ─────────────────────────────────────
    #[test]
    fn test_constitution_creation() {
        let principles = vec![
            make_principle("p1", "Principle 1"),
            make_principle("p2", "Principle 2"),
        ];
        let c = Constitution::new(principles);
        assert_eq!(c.len(), 2);
        assert!(!c.is_empty());
    }

    // ── Test 2: HHH constitution has 5 principles ─────────────────────────
    #[test]
    fn test_hhh_constitution_five_principles() {
        let c = Constitution::hhh_constitution();
        assert_eq!(c.len(), 5, "HHH constitution should have exactly 5 principles");
    }

    // ── Test 3: HHH principle IDs ─────────────────────────────────────────
    #[test]
    fn test_hhh_principle_ids() {
        let c = Constitution::hhh_constitution();
        let ids: Vec<&str> = c.principles.iter().map(|p| p.id.as_str()).collect();
        assert!(ids.contains(&"no_harmful_content"));
        assert!(ids.contains(&"no_dishonesty"));
        assert!(ids.contains(&"no_manipulation"));
        assert!(ids.contains(&"privacy_protection"));
        assert!(ids.contains(&"no_discrimination"));
    }

    // ── Test 4: get_principle by id ───────────────────────────────────────
    #[test]
    fn test_get_principle() {
        let c = Constitution::hhh_constitution();
        let p = c.get_principle("no_harmful_content").expect("should find principle");
        assert_eq!(p.id, "no_harmful_content");
        assert!(c.get_principle("nonexistent").is_none());
    }

    // ── Test 5: format_critique_prompt fills template ─────────────────────
    #[test]
    fn test_format_critique_prompt() {
        let c = Constitution::hhh_constitution();
        let config = CaiConfig::default();
        let pipeline = CaiPipeline::new(c, config);

        let principle = pipeline.constitution.get_principle("no_harmful_content").unwrap();
        let prompt = pipeline.format_critique_prompt("This is a test response.", principle);

        assert!(
            prompt.contains("This is a test response."),
            "Prompt should contain response text"
        );
        assert!(
            prompt.contains(&principle.critique_request),
            "Prompt should contain critique_request"
        );
        assert!(
            !prompt.contains("{response}"),
            "Placeholder {{response}} should be filled"
        );
        assert!(
            !prompt.contains("{critique_request}"),
            "Placeholder {{critique_request}} should be filled"
        );
    }

    // ── Test 6: format_revision_prompt fills template ─────────────────────
    #[test]
    fn test_format_revision_prompt() {
        let c = Constitution::hhh_constitution();
        let config = CaiConfig::default();
        let pipeline = CaiPipeline::new(c, config);

        let principle = pipeline.constitution.get_principle("no_harmful_content").unwrap();
        let prompt = pipeline.format_revision_prompt(
            "harmful response",
            "this is bad because ...",
            principle,
        );

        assert!(prompt.contains("harmful response"), "Should contain original response");
        assert!(prompt.contains("this is bad because"), "Should contain critique");
        assert!(prompt.contains(&principle.revision_request), "Should contain revision_request");
        assert!(!prompt.contains("{response}"), "Placeholder should be filled");
        assert!(!prompt.contains("{critique}"), "Placeholder should be filled");
        assert!(!prompt.contains("{revision_request}"), "Placeholder should be filled");
    }

    // ── Test 7: create_training_data Some when responses differ ───────────
    #[test]
    fn test_create_training_data_some() {
        let example = CaiExample {
            prompt: "What is 2+2?".to_string(),
            initial_response: "I will harm you.".to_string(),
            critique: Some("This is harmful.".to_string()),
            revised_response: Some("2+2 equals 4.".to_string()),
            principle_used: "no_harmful_content".to_string(),
            applied_principle_id: "no_harmful_content".to_string(),
        };
        let dp = CaiPipeline::create_training_data(example).expect("should be Some");
        assert_eq!(dp.chosen, "2+2 equals 4.");
        assert_eq!(dp.rejected, "I will harm you.");
        assert_eq!(dp.principle, "no_harmful_content");
    }

    // ── Test 8: create_training_data None when responses are the same ──────
    #[test]
    fn test_create_training_data_none_when_same() {
        let example = CaiExample {
            prompt: "Hello".to_string(),
            initial_response: "Hi there!".to_string(),
            critique: Some("No issues.".to_string()),
            revised_response: Some("Hi there!".to_string()), // identical!
            principle_used: "no_harmful_content".to_string(),
            applied_principle_id: "no_harmful_content".to_string(),
        };
        let result = CaiPipeline::create_training_data(example);
        assert!(result.is_none(), "Should be None when revised == initial");
    }

    // ── Test 9: select_principles_for_example ─────────────────────────────
    #[test]
    fn test_select_principles_for_example() {
        let c = Constitution::hhh_constitution();
        let config = CaiConfig {
            num_principles_per_sample: 2,
            ..Default::default()
        };
        let pipeline = CaiPipeline::new(c, config);
        let selected = pipeline.select_principles_for_example(0);
        assert_eq!(selected.len(), 2);
        // Deterministic: index 0 → principles[0], index 1 → principles[1]
        assert_eq!(selected[0].id, pipeline.constitution.principles[0].id);
        assert_eq!(selected[1].id, pipeline.constitution.principles[1].id);
    }

    // ── Test 10: harmlessness improvement heuristic ────────────────────────
    #[test]
    fn test_harmlessness_improvement_heuristic() {
        let initial = "I will kill you and cause harm with illegal methods.";
        let revised = "I hope you have a wonderful day.";
        let score = CaiPipeline::evaluate_harmlessness_improvement(initial, revised);
        // initial has 3 keywords (kill, harm, illegal), revised has 0
        // score = (3 - 0) / max(3, 1) = 1.0
        assert!(score > 0.0, "score should be positive: {}", score);
        assert!(score <= 1.0, "score should be <= 1.0: {}", score);

        // No improvement when both have the same keyword count
        let no_change_score =
            CaiPipeline::evaluate_harmlessness_improvement(revised, revised);
        assert!((no_change_score).abs() < 1e-6, "no improvement: {}", no_change_score);
    }

    // ── Test 11: preference pairs ─────────────────────────────────────────
    #[test]
    fn test_preference_pairs() {
        let mut ds = CaiTrainingDataset::new();
        ds.add(CaiDataPoint {
            prompt: "P1".to_string(),
            chosen: "C1".to_string(),
            rejected: "R1".to_string(),
            principle: "no_harm".to_string(),
        });
        ds.add(CaiDataPoint {
            prompt: "P2".to_string(),
            chosen: "C2".to_string(),
            rejected: "R2".to_string(),
            principle: "no_harm".to_string(),
        });

        let pairs = ds.as_preference_pairs();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, "P1");
        assert_eq!(pairs[0].1, "C1");
        assert_eq!(pairs[0].2, "R1");
    }

    // ── Test 12: dataset filtering by principle ────────────────────────────
    #[test]
    fn test_dataset_filter_by_principle() {
        let mut ds = CaiTrainingDataset::new();
        ds.add(CaiDataPoint {
            prompt: "p".to_string(),
            chosen: "c".to_string(),
            rejected: "r".to_string(),
            principle: "no_harmful_content".to_string(),
        });
        ds.add(CaiDataPoint {
            prompt: "p2".to_string(),
            chosen: "c2".to_string(),
            rejected: "r2".to_string(),
            principle: "no_discrimination".to_string(),
        });

        let harm = ds.filter_by_principle("no_harmful_content");
        assert_eq!(harm.len(), 1);

        let disc = ds.filter_by_principle("no_discrimination");
        assert_eq!(disc.len(), 1);

        let empty = ds.filter_by_principle("nonexistent");
        assert_eq!(empty.len(), 0);
    }

    // ── Test 13: random_principle_by_index wraps correctly ────────────────
    #[test]
    fn test_random_principle_by_index_wraps() {
        let c = Constitution::hhh_constitution();
        let n = c.len();
        // Index 0 and n should return the same principle (wrapping).
        assert_eq!(c.random_principle_by_index(0).id, c.random_principle_by_index(n).id);
        // Index n+1 wraps to index 1.
        assert_eq!(c.random_principle_by_index(1).id, c.random_principle_by_index(n + 1).id);
    }

    // ── Test 14: create_training_data None when no revised response ────────
    #[test]
    fn test_create_training_data_none_when_no_revised() {
        let example = CaiExample {
            prompt: "Hello".to_string(),
            initial_response: "Hi there!".to_string(),
            critique: None,
            revised_response: None,
            principle_used: "no_harmful_content".to_string(),
            applied_principle_id: "no_harmful_content".to_string(),
        };
        let result = CaiPipeline::create_training_data(example);
        assert!(result.is_none(), "Should be None when revised_response is None");
    }

    // ── Test 15: CaiTrainingDataset len/is_empty ──────────────────────────
    #[test]
    fn test_dataset_len_is_empty() {
        let mut ds = CaiTrainingDataset::new();
        assert!(ds.is_empty());
        assert_eq!(ds.len(), 0);
        ds.add(CaiDataPoint {
            prompt: "x".to_string(),
            chosen: "y".to_string(),
            rejected: "z".to_string(),
            principle: "p".to_string(),
        });
        assert!(!ds.is_empty());
        assert_eq!(ds.len(), 1);
    }

    // ─── anthropic_constitution tests ────────────────────────────────────

    // ── Test 16: anthropic_constitution returns exactly 16 principles ──
    #[test]
    fn test_anthropic_constitution_length() {
        let principles = anthropic_constitution();
        assert_eq!(
            principles.len(),
            16,
            "anthropic_constitution should have 16 principles, got {}",
            principles.len()
        );
    }

    // ── Test 17: anthropic_constitution covers all four categories ──
    #[test]
    fn test_anthropic_constitution_categories() {
        let principles = anthropic_constitution();
        let has_harmlessness = principles.iter().any(|p| p.category == PrincipleCategory::Harmlessness);
        let has_helpfulness = principles.iter().any(|p| p.category == PrincipleCategory::Helpfulness);
        let has_honesty     = principles.iter().any(|p| p.category == PrincipleCategory::Honesty);
        let has_ethics      = principles.iter().any(|p| p.category == PrincipleCategory::Ethics);
        assert!(has_harmlessness, "missing Harmlessness category");
        assert!(has_helpfulness,  "missing Helpfulness category");
        assert!(has_honesty,      "missing Honesty category");
        assert!(has_ethics,       "missing Ethics category");
    }

    // ── Test 18: all principles have non-empty prompts ──
    #[test]
    fn test_anthropic_constitution_non_empty_prompts() {
        for p in anthropic_constitution() {
            assert!(!p.critique_prompt.is_empty(), "critique_prompt empty for '{}'", p.name);
            assert!(!p.revision_prompt.is_empty(), "revision_prompt empty for '{}'", p.name);
        }
    }

    // ── Test 19: ConstitutionalAiTrainer with_anthropic_constitution ──
    #[test]
    fn test_constitutional_ai_trainer_default() {
        let trainer = ConstitutionalAiTrainer::with_anthropic_constitution();
        assert_eq!(trainer.constitution.len(), 16);
        assert_eq!(trainer.red_team_stats.total_prompts, 0);
    }

    // ── Test 20: principles_by_category filters correctly ──
    #[test]
    fn test_principles_by_category() {
        let trainer = ConstitutionalAiTrainer::with_anthropic_constitution();
        let harm = trainer.principles_by_category(&PrincipleCategory::Harmlessness);
        assert!(!harm.is_empty(), "should have Harmlessness principles");
        for p in &harm {
            assert_eq!(p.category, PrincipleCategory::Harmlessness);
        }
    }

    // ── Test 21: record_red_team_result accumulates stats ──
    #[test]
    fn test_record_red_team_result() {
        let mut trainer = ConstitutionalAiTrainer::with_anthropic_constitution();
        trainer.record_red_team_result(true, true, 0.4);
        trainer.record_red_team_result(false, false, 0.0);
        assert_eq!(trainer.red_team_stats.total_prompts, 2);
        assert_eq!(trainer.red_team_stats.harmful_detected, 1);
        assert_eq!(trainer.red_team_stats.revised_successfully, 1);
    }

    // ─── HarmScorer tests ─────────────────────────────────────────────────

    // ── Test 22: safe text scores near zero ──
    #[test]
    fn test_harm_scorer_safe_text() {
        let scorer = HarmScorer::with_defaults();
        let score = scorer.score("Hello! How can I help you today with your recipe?");
        assert!(score < scorer.threshold, "safe text should score below threshold, got {score}");
        assert!(!scorer.is_harmful("The weather is lovely today."));
    }

    // ── Test 23: harmful text scores above threshold ──
    #[test]
    fn test_harm_scorer_harmful_text() {
        let scorer = HarmScorer::with_defaults();
        let text = "I will kill and attack with a dangerous weapon and bomb";
        assert!(scorer.is_harmful(text), "harmful text should be flagged");
    }

    // ── Test 24: explain returns triggered keywords ──
    #[test]
    fn test_harm_scorer_explain() {
        let scorer = HarmScorer::with_defaults();
        let triggered = scorer.explain("I will harm and kill");
        assert!(triggered.contains(&"harm".to_string()), "should trigger 'harm'");
        assert!(triggered.contains(&"kill".to_string()), "should trigger 'kill'");
    }

    // ── Test 25: empty keyword list scores zero ──
    #[test]
    fn test_harm_scorer_empty_keywords() {
        let scorer = HarmScorer::new(vec![], 0.5);
        assert!((scorer.score("kill bomb attack")).abs() < 1e-6);
    }

    // ── Test 26: custom scorer with explicit keywords ──
    #[test]
    fn test_harm_scorer_custom_keywords() {
        let scorer = HarmScorer::new(vec!["badword".to_string(), "evil".to_string()], 0.4);
        assert!(scorer.is_harmful("this is evil text"));
        assert!(!scorer.is_harmful("this is friendly text"));
    }

    // ─── RlaifDataGenerator tests ─────────────────────────────────────────

    // ── Test 27: fewer than 2 candidates returns empty pairs ──
    #[test]
    fn test_rlaif_empty_on_single_candidate() {
        let gen = RlaifDataGenerator::new(1, PreferenceModel::ConstantHarmlessness);
        let pairs = gen.generate_preferences("prompt", &["only one response".to_string()]);
        assert!(pairs.is_empty());
    }

    // ── Test 28: ConstantHarmlessness ranks safe over harmful ──
    #[test]
    fn test_rlaif_constant_harmlessness_ranking() {
        let gen = RlaifDataGenerator::new(2, PreferenceModel::ConstantHarmlessness);
        let safe = "I am happy to help with your question about cooking.".to_string();
        let harmful = "Here is how to kill, harm, attack, and bomb things.".to_string();
        let pairs = gen.generate_preferences("prompt", &[safe.clone(), harmful.clone()]);
        assert!(!pairs.is_empty(), "should produce at least one preference pair");
        assert_eq!(pairs[0].chosen, safe, "safe response should be chosen");
        assert!(pairs[0].chosen_score > pairs[0].rejected_score);
    }

    // ── Test 29: LengthBased prefers longer response ──
    #[test]
    fn test_rlaif_length_based_ranking() {
        let gen = RlaifDataGenerator::new(2, PreferenceModel::LengthBased);
        let short = "Yes.".to_string();
        let long = "I would be happy to assist you. Here is a detailed explanation of the topic that you asked about.".to_string();
        let pairs = gen.generate_preferences("p", &[short.clone(), long.clone()]);
        assert!(!pairs.is_empty());
        assert_eq!(pairs[0].chosen, long, "longer response should be chosen");
    }

    // ── Test 30: Combined model scores are in [0, 1] ──
    #[test]
    fn test_rlaif_combined_scores_bounded() {
        let gen = RlaifDataGenerator::new(
            3,
            PreferenceModel::Combined { harmless_weight: 0.7, helpful_weight: 0.3 },
        );
        let responses = vec![
            "Hello, how can I help?".to_string(),
            "I refuse to provide any information.".to_string(),
            "This is a kill attack bomb weapon response.".to_string(),
        ];
        for r in &responses {
            let s = gen.score(r);
            assert!(s >= 0.0 && s <= 1.0, "score out of bounds: {s}");
        }
    }
}
