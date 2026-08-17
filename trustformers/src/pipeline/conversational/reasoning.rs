//! Advanced reasoning capabilities and context processing.

use super::types::*;
use crate::core::error::Result;

/// Advanced reasoning engine for conversational AI
#[derive(Debug)]
pub struct ReasoningEngine {
    pub config: ReasoningConfig,
}

impl ReasoningEngine {
    pub fn new(config: ReasoningConfig) -> Self {
        Self { config }
    }

    /// Process reasoning step in conversation context
    pub fn process_reasoning_step(
        &self,
        context: &mut ReasoningContext,
        input: &str,
        step_type: ReasoningType,
    ) -> Result<ReasoningStep> {
        if !self.config.enabled {
            return Ok(ReasoningStep {
                step_type,
                description: "Reasoning disabled".to_string(),
                inputs: vec![input.to_string()],
                output: input.to_string(),
                confidence: 1.0,
            });
        }

        let step = match step_type {
            ReasoningType::Logical => self.process_logical_reasoning(input, context),
            ReasoningType::Causal => self.process_causal_reasoning(input, context),
            ReasoningType::Analogical => self.process_analogical_reasoning(input, context),
            ReasoningType::Creative => self.process_creative_reasoning(input, context),
            ReasoningType::Mathematical => self.process_mathematical_reasoning(input, context),
            ReasoningType::Emotional => self.process_emotional_reasoning(input, context),
        }?;

        context.reasoning_chain.push(step.clone());
        context.confidence = self.update_context_confidence(context);

        Ok(step)
    }

    fn process_logical_reasoning(
        &self,
        input: &str,
        context: &ReasoningContext,
    ) -> Result<ReasoningStep> {
        let mut premises = Vec::new();
        let mut conclusion = String::new();

        // Simple logical pattern detection
        if input.contains("if") && input.contains("then") {
            let parts: Vec<&str> = input.split("then").collect();
            if parts.len() == 2 {
                premises.push(parts[0].trim().to_string());
                conclusion = parts[1].trim().to_string();
            }
        } else if input.contains("because") {
            let parts: Vec<&str> = input.split("because").collect();
            if parts.len() == 2 {
                conclusion = parts[0].trim().to_string();
                premises.push(parts[1].trim().to_string());
            }
        } else {
            // Fallback: treat input as premise for logical chain
            premises.push(input.to_string());
            conclusion = self.derive_logical_conclusion(&premises, context)?;
        }

        Ok(ReasoningStep {
            step_type: ReasoningType::Logical,
            description: "Applied logical reasoning to derive conclusion".to_string(),
            inputs: premises,
            output: conclusion,
            confidence: 0.8,
        })
    }

    fn process_causal_reasoning(
        &self,
        input: &str,
        context: &ReasoningContext,
    ) -> Result<ReasoningStep> {
        let mut causes = Vec::new();
        let mut effects = Vec::new();

        // Detect causal language patterns
        if input.contains("causes") || input.contains("leads to") {
            let cause_effect: Vec<&str> = if input.contains("causes") {
                input.split("causes").collect()
            } else {
                input.split("leads to").collect()
            };
            if cause_effect.len() >= 2 {
                causes.push(cause_effect[0].trim().to_string());
                effects.push(cause_effect[1].trim().to_string());
            }
        } else if input.contains("results in") {
            let parts: Vec<&str> = input.split("results in").collect();
            if parts.len() >= 2 {
                causes.push(parts[0].trim().to_string());
                effects.push(parts[1].trim().to_string());
            }
        } else {
            // Analyze for implicit causal relationships
            causes.push(input.to_string());
            effects.push(self.predict_effects(&causes, context)?);
        }

        let output = format!(
            "Causal relationship: {} -> {}",
            causes.join(", "),
            effects.join(", ")
        );

        Ok(ReasoningStep {
            step_type: ReasoningType::Causal,
            description: "Analyzed causal relationships".to_string(),
            inputs: causes,
            output,
            confidence: 0.7,
        })
    }

    fn process_analogical_reasoning(
        &self,
        input: &str,
        context: &ReasoningContext,
    ) -> Result<ReasoningStep> {
        let analogies = self.find_analogies(input, context)?;

        let output = if analogies.is_empty() {
            format!("Analyzed '{}' for analogical patterns", input)
        } else {
            format!("Found analogies: {}", analogies.join("; "))
        };

        Ok(ReasoningStep {
            step_type: ReasoningType::Analogical,
            description: "Searched for analogical relationships".to_string(),
            inputs: vec![input.to_string()],
            output,
            confidence: 0.6,
        })
    }

    fn process_creative_reasoning(
        &self,
        input: &str,
        context: &ReasoningContext,
    ) -> Result<ReasoningStep> {
        let creative_elements = self.extract_creative_elements(input)?;
        let output = self.generate_creative_response(input, &creative_elements, context)?;

        Ok(ReasoningStep {
            step_type: ReasoningType::Creative,
            description: "Applied creative reasoning and idea generation".to_string(),
            inputs: vec![input.to_string()],
            output,
            confidence: 0.7,
        })
    }

    fn process_mathematical_reasoning(
        &self,
        input: &str,
        context: &ReasoningContext,
    ) -> Result<ReasoningStep> {
        let math_expressions = self.extract_mathematical_expressions(input)?;
        let solutions = self.solve_mathematical_problems(&math_expressions)?;

        let output = if solutions.is_empty() {
            format!("Analyzed '{}' for mathematical content", input)
        } else {
            format!("Mathematical solutions: {}", solutions.join("; "))
        };

        Ok(ReasoningStep {
            step_type: ReasoningType::Mathematical,
            description: "Applied mathematical reasoning".to_string(),
            inputs: math_expressions,
            output,
            confidence: 0.9,
        })
    }

    fn process_emotional_reasoning(
        &self,
        input: &str,
        context: &ReasoningContext,
    ) -> Result<ReasoningStep> {
        let emotional_content = self.analyze_emotional_content(input)?;
        let empathetic_response = self.generate_empathetic_response(input, &emotional_content)?;

        Ok(ReasoningStep {
            step_type: ReasoningType::Emotional,
            description: "Applied emotional reasoning and empathy".to_string(),
            inputs: vec![input.to_string()],
            output: empathetic_response,
            confidence: 0.8,
        })
    }

    /// Derive what actually follows from `premises` by forward chaining.
    ///
    /// Conditionals ("if A then B", "A implies B") are turned into rules and
    /// combined with the asserted facts using modus ponens and modus tollens,
    /// drawing on facts already established earlier in `context`. When no rule
    /// fires the result says so — it does not restate the premises as though
    /// something had been concluded.
    fn derive_logical_conclusion(
        &self,
        premises: &[String],
        context: &ReasoningContext,
    ) -> Result<String> {
        if premises.is_empty() {
            return Ok("No premises available for logical conclusion".to_string());
        }

        let mut rules: Vec<(String, String)> = Vec::new();
        let mut facts: Vec<String> = Vec::new();

        // Facts established by earlier logical steps are available as premises.
        for step in context
            .reasoning_chain
            .iter()
            .filter(|step| matches!(step.step_type, ReasoningType::Logical))
        {
            match parse_conditional(&step.output) {
                Some(rule) => rules.push(rule),
                None => facts.push(normalize_proposition(&step.output)),
            }
        }

        for premise in premises {
            match parse_conditional(premise) {
                Some(rule) => rules.push(rule),
                None => facts.push(normalize_proposition(premise)),
            }
        }

        if rules.is_empty() {
            return Ok(format!(
                "No conditional premise was supplied, so nothing follows from: {}",
                premises.join(", ")
            ));
        }

        // Forward chaining: keep applying rules until the fact set stops growing.
        let mut derived: Vec<String> = Vec::new();
        let mut changed = true;
        while changed {
            changed = false;
            for (antecedent, consequent) in &rules {
                // Modus ponens: antecedent holds, therefore the consequent does.
                if facts.iter().any(|f| propositions_match(f, antecedent))
                    && !facts.iter().any(|f| propositions_match(f, consequent))
                {
                    facts.push(consequent.clone());
                    derived.push(format!("{consequent} (modus ponens from \"{antecedent}\")"));
                    changed = true;
                }
                // Modus tollens: the consequent is denied, therefore so is the
                // antecedent.
                let denied_consequent = format!("not {consequent}");
                let denied_antecedent = format!("not {antecedent}");
                if facts.iter().any(|f| propositions_match(f, &denied_consequent))
                    && !facts.iter().any(|f| propositions_match(f, &denied_antecedent))
                {
                    facts.push(denied_antecedent.clone());
                    derived.push(format!(
                        "{denied_antecedent} (modus tollens from \"{consequent}\")"
                    ));
                    changed = true;
                }
            }
        }

        if derived.is_empty() {
            Ok(format!(
                "No rule applies: none of the {} premise(s) satisfies the antecedent of the \
                 {} conditional(s)",
                premises.len(),
                rules.len()
            ))
        } else {
            Ok(format!("Derived: {}", derived.join("; ")))
        }
    }

    /// Chain stated cause→effect relations forward from `causes`.
    ///
    /// Relations come from the causal steps already in `context`; a cause with
    /// no matching relation is reported as such instead of being echoed back as
    /// a "predicted effect".
    fn predict_effects(&self, causes: &[String], context: &ReasoningContext) -> Result<String> {
        if causes.is_empty() {
            return Ok("No causes specified for effect prediction".to_string());
        }

        // Collect the causal relations this conversation has established.
        let mut relations: Vec<(String, String)> = Vec::new();
        for step in context
            .reasoning_chain
            .iter()
            .filter(|step| matches!(step.step_type, ReasoningType::Causal))
        {
            if let Some(relation) = parse_causal_relation(&step.output) {
                relations.push(relation);
            }
        }

        if relations.is_empty() {
            return Ok(format!(
                "No causal relation has been established for: {}",
                causes.join(", ")
            ));
        }

        // Transitive closure over the known relations.
        let mut frontier: Vec<String> = causes.iter().map(|c| normalize_proposition(c)).collect();
        let mut reached: Vec<String> = Vec::new();
        let mut guard = 0usize;
        while let Some(current) = frontier.pop() {
            guard += 1;
            if guard > 64 {
                break;
            }
            for (cause, effect) in &relations {
                if propositions_match(&current, cause)
                    && !reached.iter().any(|e| propositions_match(e, effect))
                {
                    reached.push(effect.clone());
                    frontier.push(effect.clone());
                }
            }
        }

        if reached.is_empty() {
            Ok(format!(
                "None of the {} known causal relation(s) applies to: {}",
                relations.len(),
                causes.join(", ")
            ))
        } else {
            Ok(format!("Follows causally: {}", reached.join(" -> ")))
        }
    }

    fn find_analogies(&self, input: &str, context: &ReasoningContext) -> Result<Vec<String>> {
        let mut analogies = Vec::new();

        // Simple analogy detection patterns
        if input.contains("like") || input.contains("similar to") {
            let parts: Vec<&str> = if input.contains("like") {
                input.split("like").collect()
            } else {
                input.split("similar to").collect()
            };
            if parts.len() >= 2 {
                analogies.push(format!("{} is like {}", parts[0].trim(), parts[1].trim()));
            }
        }

        // Check against previous reasoning steps for analogical patterns
        for step in &context.reasoning_chain {
            if let ReasoningType::Analogical = step.step_type {
                // Build on previous analogies
                if step.output.contains("analogy") {
                    analogies.push(format!("Related to previous analogy: {}", step.output));
                }
            }
        }

        Ok(analogies)
    }

    fn extract_creative_elements(&self, input: &str) -> Result<Vec<String>> {
        let mut elements = Vec::new();

        let creative_keywords = [
            "imagine",
            "creative",
            "innovative",
            "original",
            "unique",
            "novel",
            "inventive",
            "artistic",
            "design",
            "create",
        ];

        for keyword in &creative_keywords {
            if input.to_lowercase().contains(keyword) {
                elements.push(keyword.to_string());
            }
        }

        Ok(elements)
    }

    fn generate_creative_response(
        &self,
        input: &str,
        elements: &[String],
        context: &ReasoningContext,
    ) -> Result<String> {
        if elements.is_empty() {
            return Ok(format!("Applied creative analysis to: {}", input));
        }

        Ok(format!(
            "Creative exploration of '{}' focusing on: {}",
            input,
            elements.join(", ")
        ))
    }

    fn extract_mathematical_expressions(&self, input: &str) -> Result<Vec<String>> {
        let mut expressions = Vec::new();

        // Simple math pattern detection
        let math_patterns = [
            r"\d+\s*[+\-*/]\s*\d+",
            r"\d+\s*=\s*\d+",
            r"calculate|compute|solve",
            r"\d+%",
            r"\$\d+",
        ];

        for pattern in &math_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                for mat in regex.find_iter(input) {
                    expressions.push(mat.as_str().to_string());
                }
            }
        }

        Ok(expressions)
    }

    fn solve_mathematical_problems(&self, expressions: &[String]) -> Result<Vec<String>> {
        let mut solutions = Vec::new();

        for expr in expressions {
            // Very basic mathematical problem solving
            if expr.contains('+') {
                solutions.push(format!("Addition problem: {}", expr));
            } else if expr.contains('-') {
                solutions.push(format!("Subtraction problem: {}", expr));
            } else if expr.contains('*') {
                solutions.push(format!("Multiplication problem: {}", expr));
            } else if expr.contains('/') {
                solutions.push(format!("Division problem: {}", expr));
            } else {
                solutions.push(format!("Mathematical expression: {}", expr));
            }
        }

        Ok(solutions)
    }

    fn analyze_emotional_content(&self, input: &str) -> Result<EmotionalContent> {
        let mut emotions = Vec::new();
        let mut intensity: f32 = 0.0;

        let emotion_patterns = [
            (
                "joy",
                &["happy", "joyful", "excited", "cheerful", "delighted"] as &[&str],
            ),
            (
                "sadness",
                &["sad", "depressed", "sorrowful", "melancholy", "grief"],
            ),
            (
                "anger",
                &["angry", "furious", "irritated", "annoyed", "rage"],
            ),
            (
                "fear",
                &["afraid", "scared", "anxious", "worried", "nervous"],
            ),
            (
                "surprise",
                &["surprised", "amazed", "astonished", "shocked"],
            ),
            ("love", &["love", "affection", "adore", "cherish", "care"]),
        ];

        let content_lower = input.to_lowercase();

        for (emotion, keywords) in &emotion_patterns {
            for keyword in *keywords {
                if content_lower.contains(keyword) {
                    emotions.push(emotion.to_string());
                    intensity += 0.2;
                    break;
                }
            }
        }

        // Analyze intensity indicators
        if input.contains('!') {
            intensity += 0.3;
        }
        if input.chars().filter(|c| c.is_uppercase()).count() > input.len() / 3 {
            intensity += 0.2;
        }

        Ok(EmotionalContent {
            detected_emotions: emotions,
            intensity: intensity.min(1.0),
            valence: self.calculate_emotional_valence(&content_lower),
        })
    }

    fn calculate_emotional_valence(&self, content: &str) -> f32 {
        let positive_words = ["good", "great", "wonderful", "amazing", "love", "happy"];
        let negative_words = ["bad", "terrible", "awful", "hate", "sad", "angry"];

        let pos_count = positive_words.iter().filter(|word| content.contains(*word)).count();
        let neg_count = negative_words.iter().filter(|word| content.contains(*word)).count();

        if pos_count > neg_count {
            0.7
        } else if neg_count > pos_count {
            -0.7
        } else {
            0.0
        }
    }

    fn generate_empathetic_response(
        &self,
        input: &str,
        emotional_content: &EmotionalContent,
    ) -> Result<String> {
        if emotional_content.detected_emotions.is_empty() {
            return Ok(format!("Acknowledged emotional context in: {}", input));
        }

        let response = match emotional_content.valence {
            v if v > 0.5 => {
                format!(
                    "I can sense the positive emotions ({}) in your message",
                    emotional_content.detected_emotions.join(", ")
                )
            },
            v if v < -0.5 => {
                format!(
                    "I understand this might be difficult given the emotions you're experiencing ({})",
                    emotional_content.detected_emotions.join(", ")
                )
            },
            _ => {
                format!(
                    "I recognize the emotional complexity in your message involving {}",
                    emotional_content.detected_emotions.join(", ")
                )
            },
        };

        Ok(response)
    }

    fn update_context_confidence(&self, context: &ReasoningContext) -> f32 {
        if context.reasoning_chain.is_empty() {
            return 1.0;
        }

        let total_confidence: f32 =
            context.reasoning_chain.iter().map(|step| step.confidence).sum();

        let avg_confidence = total_confidence / context.reasoning_chain.len() as f32;

        // Adjust based on chain length and consistency
        let chain_factor = if context.reasoning_chain.len() > 5 {
            0.9 // Longer chains might have accumulated uncertainty
        } else {
            1.0
        };

        (avg_confidence * chain_factor).min(1.0)
    }

    /// Detect the primary reasoning type needed for input
    pub fn detect_reasoning_type(&self, input: &str) -> ReasoningType {
        let content_lower = input.to_lowercase();

        // Mathematical patterns
        if content_lower.contains("calculate")
            || content_lower.contains("math")
            || regex::Regex::new(r"\d+\s*[+\-*/]\s*\d+")
                .map(|re| re.is_match(&content_lower))
                .unwrap_or(false)
        {
            return ReasoningType::Mathematical;
        }

        // Logical patterns
        if content_lower.contains("because")
            || content_lower.contains("therefore")
            || content_lower.contains("if") && content_lower.contains("then")
        {
            return ReasoningType::Logical;
        }

        // Causal patterns
        if content_lower.contains("causes")
            || content_lower.contains("leads to")
            || content_lower.contains("results in")
        {
            return ReasoningType::Causal;
        }

        // Creative patterns
        if content_lower.contains("imagine")
            || content_lower.contains("creative")
            || content_lower.contains("design")
        {
            return ReasoningType::Creative;
        }

        // Emotional patterns
        if content_lower.contains("feel")
            || content_lower.contains("emotion")
            || ["happy", "sad", "angry", "afraid"]
                .iter()
                .any(|&word| content_lower.contains(word))
        {
            return ReasoningType::Emotional;
        }

        // Analogical patterns
        if content_lower.contains("like")
            || content_lower.contains("similar")
            || content_lower.contains("analogous")
        {
            return ReasoningType::Analogical;
        }

        // Default to logical reasoning
        ReasoningType::Logical
    }

    /// Check if reasoning timeout is exceeded
    pub fn is_timeout_exceeded(&self, start_time: std::time::Instant) -> bool {
        start_time.elapsed().as_millis() > self.config.timeout_ms as u128
    }

    /// Generate reasoning summary
    pub fn generate_reasoning_summary(&self, context: &ReasoningContext) -> ReasoningSummary {
        let step_types: std::collections::HashMap<ReasoningType, usize> = context
            .reasoning_chain
            .iter()
            .fold(std::collections::HashMap::new(), |mut acc, step| {
                *acc.entry(step.step_type.clone()).or_insert(0) += 1;
                acc
            });

        let avg_confidence = if context.reasoning_chain.is_empty() {
            0.0
        } else {
            context.reasoning_chain.iter().map(|s| s.confidence).sum::<f32>()
                / context.reasoning_chain.len() as f32
        };

        ReasoningSummary {
            total_steps: context.reasoning_chain.len(),
            step_type_distribution: step_types,
            avg_confidence,
            final_confidence: context.confidence,
            current_goal: context.current_goal.clone(),
            evidence_count: context.evidence.len(),
            assumption_count: context.assumptions.len(),
        }
    }
}

/// Emotional content analysis
#[derive(Debug, Clone)]
pub struct EmotionalContent {
    pub detected_emotions: Vec<String>,
    pub intensity: f32,
    pub valence: f32, // -1.0 (negative) to 1.0 (positive)
}

/// Summary of reasoning process
#[derive(Debug, Clone)]
pub struct ReasoningSummary {
    pub total_steps: usize,
    pub step_type_distribution: std::collections::HashMap<ReasoningType, usize>,
    pub avg_confidence: f32,
    pub final_confidence: f32,
    pub current_goal: Option<String>,
    pub evidence_count: usize,
    pub assumption_count: usize,
}

/// Normalise a proposition for comparison: trimmed, lower-cased, without
/// trailing punctuation or a leading connective.
fn normalize_proposition(text: &str) -> String {
    let lowered = text.trim().to_lowercase();
    let stripped = lowered
        .trim_start_matches("therefore ")
        .trim_start_matches("so ")
        .trim_start_matches("thus ")
        .trim_start_matches("then ")
        .trim_end_matches(['.', '!', '?', ',', ';'])
        .trim();
    stripped.to_string()
}

/// Two propositions match when their normalised forms are equal, or when one
/// fully contains the other (handling "it rains" vs "it rains today").
fn propositions_match(a: &str, b: &str) -> bool {
    let (a, b) = (normalize_proposition(a), normalize_proposition(b));
    if a.is_empty() || b.is_empty() {
        return false;
    }
    a == b || a.contains(&b) || b.contains(&a)
}

/// Parse a conditional premise into `(antecedent, consequent)`.
///
/// Recognises "if A then B", "if A, B", and "A implies B".
fn parse_conditional(text: &str) -> Option<(String, String)> {
    let lowered = text.trim().to_lowercase();
    if let Some(rest) = lowered.strip_prefix("if ") {
        if let Some((antecedent, consequent)) = rest.split_once(" then ") {
            return Some((
                normalize_proposition(antecedent),
                normalize_proposition(consequent),
            ));
        }
        if let Some((antecedent, consequent)) = rest.split_once(", ") {
            return Some((
                normalize_proposition(antecedent),
                normalize_proposition(consequent),
            ));
        }
        return None;
    }
    for marker in [" implies ", " entails "] {
        if let Some((antecedent, consequent)) = lowered.split_once(marker) {
            return Some((
                normalize_proposition(antecedent),
                normalize_proposition(consequent),
            ));
        }
    }
    None
}

/// Parse a causal statement into `(cause, effect)`.
fn parse_causal_relation(text: &str) -> Option<(String, String)> {
    let lowered = text.trim().to_lowercase();
    // Statements produced by `process_causal_reasoning` use "A -> B".
    if let Some((cause, effect)) = lowered
        .strip_prefix("causal relationship: ")
        .and_then(|rest| rest.split_once(" -> "))
    {
        return Some((normalize_proposition(cause), normalize_proposition(effect)));
    }
    for marker in [" causes ", " leads to ", " results in ", " -> "] {
        if let Some((cause, effect)) = lowered.split_once(marker) {
            return Some((normalize_proposition(cause), normalize_proposition(effect)));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Regression: `derive_logical_conclusion` and `predict_effects` used to
    // return echo templates ("Logical conclusion derived from: …",
    // "Predicted effects based on causes: …") without performing any
    // inference. These assert real derivation.
    // -----------------------------------------------------------------------

    #[test]
    fn modus_ponens_actually_fires() {
        let engine = ReasoningEngine::new(default_config());
        let ctx = make_context();
        let conclusion = engine
            .derive_logical_conclusion(
                &[
                    "if it rains then the ground is wet".to_string(),
                    "it rains".to_string(),
                ],
                &ctx,
            )
            .expect("derivation should succeed");
        assert!(
            conclusion.contains("the ground is wet"),
            "modus ponens must produce the consequent, got: {conclusion}"
        );
        assert!(
            !conclusion.starts_with("Logical conclusion derived from"),
            "the old echo template must be gone: {conclusion}"
        );
    }

    #[test]
    fn no_rule_means_no_conclusion_is_claimed() {
        let engine = ReasoningEngine::new(default_config());
        let ctx = make_context();
        let conclusion = engine
            .derive_logical_conclusion(&["the sky is blue".to_string()], &ctx)
            .expect("derivation should succeed");
        assert!(
            conclusion.contains("No conditional premise"),
            "without a rule nothing follows, got: {conclusion}"
        );
    }

    #[test]
    fn effects_require_an_established_relation() {
        let engine = ReasoningEngine::new(default_config());
        let mut ctx = make_context();
        let unrelated = engine
            .predict_effects(&["heavy rain".to_string()], &ctx)
            .expect("prediction should succeed");
        assert!(
            unrelated.contains("No causal relation"),
            "with no relations nothing may be predicted, got: {unrelated}"
        );

        ctx.reasoning_chain.push(ReasoningStep {
            step_type: ReasoningType::Causal,
            description: "known relation".to_string(),
            inputs: vec!["heavy rain".to_string()],
            output: "Causal relationship: heavy rain -> flooding".to_string(),
            confidence: 0.7,
        });
        let derived = engine
            .predict_effects(&["heavy rain".to_string()], &ctx)
            .expect("prediction should succeed");
        assert!(
            derived.contains("flooding"),
            "the established relation must be followed, got: {derived}"
        );
    }

    fn make_context() -> ReasoningContext {
        ReasoningContext {
            reasoning_chain: Vec::new(),
            current_goal: None,
            evidence: Vec::new(),
            assumptions: Vec::new(),
            confidence: 1.0,
        }
    }

    fn default_config() -> ReasoningConfig {
        ReasoningConfig::default()
    }

    // ---- Engine construction ----

    #[test]
    fn test_engine_new_stores_config() {
        let config = default_config();
        let engine = ReasoningEngine::new(config.clone());
        assert_eq!(
            engine.config.enabled, config.enabled,
            "config must be stored correctly"
        );
    }

    // ---- detect_reasoning_type ----

    #[test]
    fn test_detect_reasoning_type_mathematical() {
        let engine = ReasoningEngine::new(default_config());
        let rt = engine.detect_reasoning_type("please calculate the sum of 3 + 5");
        assert_eq!(
            rt,
            ReasoningType::Mathematical,
            "'calculate' should detect Mathematical"
        );
    }

    #[test]
    fn test_detect_reasoning_type_logical_because() {
        let engine = ReasoningEngine::new(default_config());
        let rt = engine.detect_reasoning_type("It is cold because winter arrived.");
        assert_eq!(
            rt,
            ReasoningType::Logical,
            "'because' should detect Logical"
        );
    }

    #[test]
    fn test_detect_reasoning_type_causal() {
        let engine = ReasoningEngine::new(default_config());
        let rt = engine.detect_reasoning_type("Exercise leads to better health.");
        assert_eq!(rt, ReasoningType::Causal, "'leads to' should detect Causal");
    }

    #[test]
    fn test_detect_reasoning_type_creative() {
        let engine = ReasoningEngine::new(default_config());
        let rt = engine.detect_reasoning_type("Imagine a creative design for this project.");
        assert_eq!(
            rt,
            ReasoningType::Creative,
            "'imagine' and 'creative' should detect Creative"
        );
    }

    #[test]
    fn test_detect_reasoning_type_emotional() {
        let engine = ReasoningEngine::new(default_config());
        let rt = engine.detect_reasoning_type("I feel happy today.");
        assert_eq!(
            rt,
            ReasoningType::Emotional,
            "'feel' should detect Emotional"
        );
    }

    #[test]
    fn test_detect_reasoning_type_default_logical_fallback() {
        let engine = ReasoningEngine::new(default_config());
        let rt = engine.detect_reasoning_type("The earth orbits the sun.");
        // No special keyword → defaults to Logical
        assert_eq!(
            rt,
            ReasoningType::Logical,
            "generic text should default to Logical"
        );
    }

    // ---- process_reasoning_step ----

    #[test]
    fn test_process_logical_step_returns_step() {
        let engine = ReasoningEngine::new(default_config());
        let mut ctx = make_context();
        let result = engine.process_reasoning_step(&mut ctx, "if A then B", ReasoningType::Logical);
        assert!(result.is_ok(), "logical reasoning step must succeed");
        let step = result.expect("step must be Ok");
        assert_eq!(
            step.step_type,
            ReasoningType::Logical,
            "step type must be Logical"
        );
    }

    #[test]
    fn test_process_step_appends_to_chain() {
        let engine = ReasoningEngine::new(default_config());
        let mut ctx = make_context();
        let _ = engine
            .process_reasoning_step(&mut ctx, "something causes another", ReasoningType::Causal)
            .expect("step must succeed");
        assert_eq!(
            ctx.reasoning_chain.len(),
            1,
            "step must be appended to reasoning chain"
        );
    }

    #[test]
    fn test_process_multiple_steps_accumulate() {
        let engine = ReasoningEngine::new(default_config());
        let mut ctx = make_context();
        for _ in 0..3u32 {
            engine
                .process_reasoning_step(&mut ctx, "input", ReasoningType::Logical)
                .expect("step must succeed");
        }
        assert_eq!(
            ctx.reasoning_chain.len(),
            3,
            "three steps must accumulate in chain"
        );
    }

    #[test]
    fn test_process_step_disabled_engine_returns_passthrough() {
        let mut config = default_config();
        config.enabled = false;
        let engine = ReasoningEngine::new(config);
        let mut ctx = make_context();
        let step = engine
            .process_reasoning_step(&mut ctx, "test input", ReasoningType::Logical)
            .expect("disabled engine should still return a step");
        assert_eq!(
            step.output, "test input",
            "disabled engine must passthrough input as output"
        );
    }

    // ---- generate_reasoning_summary ----

    #[test]
    fn test_summary_empty_chain() {
        let engine = ReasoningEngine::new(default_config());
        let ctx = make_context();
        let summary = engine.generate_reasoning_summary(&ctx);
        assert_eq!(
            summary.total_steps, 0,
            "empty chain should yield 0 total_steps"
        );
        assert!(
            (summary.avg_confidence - 0.0).abs() < f32::EPSILON,
            "empty chain avg confidence should be 0"
        );
    }

    #[test]
    fn test_summary_reflects_step_count() {
        let engine = ReasoningEngine::new(default_config());
        let mut ctx = make_context();
        for _ in 0..4u32 {
            engine
                .process_reasoning_step(&mut ctx, "premise", ReasoningType::Logical)
                .expect("step must succeed");
        }
        let summary = engine.generate_reasoning_summary(&ctx);
        assert_eq!(
            summary.total_steps, 4,
            "summary total_steps must reflect chain length"
        );
    }

    #[test]
    fn test_summary_avg_confidence_within_bounds() {
        let engine = ReasoningEngine::new(default_config());
        let mut ctx = make_context();
        engine
            .process_reasoning_step(&mut ctx, "a therefore b", ReasoningType::Logical)
            .expect("step must succeed");
        let summary = engine.generate_reasoning_summary(&ctx);
        assert!(
            (0.0..=1.0).contains(&summary.avg_confidence),
            "avg_confidence must be in [0,1]"
        );
    }

    #[test]
    fn test_summary_goal_propagated() {
        let engine = ReasoningEngine::new(default_config());
        let ctx = ReasoningContext {
            reasoning_chain: Vec::new(),
            current_goal: Some("find answer".to_string()),
            evidence: Vec::new(),
            assumptions: Vec::new(),
            confidence: 1.0,
        };
        let summary = engine.generate_reasoning_summary(&ctx);
        assert_eq!(
            summary.current_goal.as_deref(),
            Some("find answer"),
            "summary must propagate the current goal"
        );
    }

    // ---- is_timeout_exceeded ----

    #[test]
    fn test_timeout_not_exceeded_immediately() {
        let engine = ReasoningEngine::new(default_config());
        let start = std::time::Instant::now();
        assert!(
            !engine.is_timeout_exceeded(start),
            "timeout must not be exceeded immediately after start"
        );
    }
}
