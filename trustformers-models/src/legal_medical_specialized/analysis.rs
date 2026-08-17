//! Document/citation/compliance heuristics for [`LegalMedicalForCausalLM`].
//!
//! These are keyword- and pattern-based classifiers, not model inference -
//! they were already real (no fabricated results) before this pass and are
//! relocated here unchanged, purely to keep the main module file under the
//! workspace's line-count limit once real attention/generation/redaction
//! were added to it.

use super::{LegalMedicalDomain, LegalMedicalForCausalLM};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Document analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentAnalysis {
    pub document_type: String,
    pub domain_classification: LegalMedicalDomain,
    pub privacy_sensitive_sections: Vec<String>,
    pub citation_count: usize,
    pub compliance_score: f32,
    pub key_entities: Vec<String>,
    pub redaction_suggestions: Vec<String>,
}

/// Citation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub citation_type: CitationType,
    pub text: String,
    pub page_number: Option<u32>,
    pub authority: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CitationType {
    LegalCase,
    Statute,
    Regulation,
    MedicalJournal,
    ClinicalTrial,
    DrugLabel,
}

/// Compliance checking report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    pub overall_score: f32,
    pub privacy_compliance: bool,
    pub regulatory_compliance: bool,
    pub violations: Vec<ComplianceViolation>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    pub violation_type: String,
    pub severity: String,
    pub location: String,
    pub description: String,
    pub suggested_fix: String,
}

impl LegalMedicalForCausalLM {
    pub fn analyze_document(&self, text: &str) -> Result<DocumentAnalysis> {
        // Analyze legal/medical documents
        let domain_classification = self.classify_domain(text)?;
        let privacy_sensitive_sections = self.identify_sensitive_sections(text)?;
        let citation_count = self.count_citations(text)?;
        let compliance_score = self.calculate_compliance_score(text)?;
        let key_entities = self.extract_key_entities(text)?;
        let redaction_suggestions = self.generate_redaction_suggestions(text)?;

        let document_type = self.classify_document_type(text)?;

        Ok(DocumentAnalysis {
            document_type,
            domain_classification,
            privacy_sensitive_sections,
            citation_count,
            compliance_score,
            key_entities,
            redaction_suggestions,
        })
    }

    pub fn extract_citations(&self, text: &str) -> Result<Vec<Citation>> {
        // Extract legal citations or medical references
        let mut citations = Vec::new();

        // Extract legal case citations
        citations.extend(self.extract_legal_case_citations(text)?);

        // Extract statute citations
        citations.extend(self.extract_statute_citations(text)?);

        // Extract medical journal citations
        citations.extend(self.extract_medical_journal_citations(text)?);

        // Extract clinical trial citations
        citations.extend(self.extract_clinical_trial_citations(text)?);

        Ok(citations)
    }

    pub fn compliance_check(&self, text: &str) -> Result<ComplianceReport> {
        // Check document for regulatory compliance
        let mut violations = Vec::new();
        let mut recommendations = Vec::new();

        // Check for HIPAA compliance (medical)
        if self.is_medical_domain() {
            violations.extend(self.check_hipaa_compliance(text)?);
        }

        // Check for GDPR compliance (general)
        violations.extend(self.check_gdpr_compliance(text)?);

        // Check for attorney-client privilege (legal)
        if self.is_legal_domain() {
            violations.extend(self.check_attorney_client_privilege(text)?);
        }

        // Generate recommendations
        recommendations.extend(self.generate_compliance_recommendations(&violations)?);

        let privacy_compliance = !violations.iter().any(|v| v.violation_type.contains("privacy"));
        let regulatory_compliance =
            !violations.iter().any(|v| v.violation_type.contains("regulatory"));

        let overall_score = if violations.is_empty() {
            1.0
        } else {
            1.0 - (violations.len() as f32 * 0.1).min(1.0)
        };

        Ok(ComplianceReport {
            overall_score,
            privacy_compliance,
            regulatory_compliance,
            violations,
            recommendations,
        })
    }

    fn classify_domain(&self, text: &str) -> Result<LegalMedicalDomain> {
        let text_lower = text.to_lowercase();

        // Medical keywords
        if text_lower.contains("patient")
            || text_lower.contains("medical")
            || text_lower.contains("diagnosis")
        {
            if text_lower.contains("clinical") || text_lower.contains("treatment") {
                Ok(LegalMedicalDomain::MedicalClinical)
            } else if text_lower.contains("research") || text_lower.contains("study") {
                Ok(LegalMedicalDomain::MedicalResearch)
            } else if text_lower.contains("drug") || text_lower.contains("medication") {
                Ok(LegalMedicalDomain::MedicalPharmacology)
            } else {
                Ok(LegalMedicalDomain::Medical)
            }
        }
        // Legal keywords
        else if text_lower.contains("court")
            || text_lower.contains("legal")
            || text_lower.contains("contract")
        {
            if text_lower.contains("contract") || text_lower.contains("agreement") {
                Ok(LegalMedicalDomain::LegalContract)
            } else if text_lower.contains("litigation") || text_lower.contains("lawsuit") {
                Ok(LegalMedicalDomain::LegalLitigation)
            } else if text_lower.contains("regulation") || text_lower.contains("compliance") {
                Ok(LegalMedicalDomain::LegalRegulatory)
            } else {
                Ok(LegalMedicalDomain::Legal)
            }
        } else {
            Ok(LegalMedicalDomain::Legal) // Default
        }
    }

    fn identify_sensitive_sections(&self, text: &str) -> Result<Vec<String>> {
        let mut sensitive_sections = Vec::new();

        // Look for patterns that might contain sensitive information
        if text.contains("SSN") || text.contains("Social Security") {
            sensitive_sections.push("Social Security Number".to_string());
        }
        if text.contains("DOB") || text.contains("Date of Birth") {
            sensitive_sections.push("Date of Birth".to_string());
        }
        if text.contains('@') && text.contains('.') {
            sensitive_sections.push("Email Address".to_string());
        }

        Ok(sensitive_sections)
    }

    fn count_citations(&self, text: &str) -> Result<usize> {
        // Simple citation counting
        let mut count = 0;

        // Legal citations (e.g., "v." for versus)
        count += text.matches(" v. ").count();
        count += text.matches(" vs. ").count();

        // Medical citations (e.g., journal references)
        count += text.matches("et al.").count();
        count += text.matches("DOI:").count();

        Ok(count)
    }

    fn calculate_compliance_score(&self, text: &str) -> Result<f32> {
        let mut score = 1.0;

        // Deduct points for potential compliance issues
        if self.contains_sensitive_info(text)? {
            score -= 0.3;
        }

        if text.to_lowercase().contains("confidential")
            && !text.to_lowercase().contains("privilege")
        {
            score -= 0.2;
        }

        Ok(f32::max(score, 0.0))
    }

    fn extract_key_entities(&self, text: &str) -> Result<Vec<String>> {
        let mut entities = Vec::new();

        // Extract potential person names (simplified)
        let words: Vec<&str> = text.split_whitespace().collect();
        for window in words.windows(2) {
            if window[0].chars().next().unwrap_or('a').is_uppercase()
                && window[1].chars().next().unwrap_or('a').is_uppercase()
            {
                entities.push(format!("{} {}", window[0], window[1]));
            }
        }

        // Extract organizations (simplified)
        if text.contains("Inc.") || text.contains("Corp.") || text.contains("LLC") {
            entities.push("Organization".to_string());
        }

        Ok(entities)
    }

    fn generate_redaction_suggestions(&self, text: &str) -> Result<Vec<String>> {
        let mut suggestions = Vec::new();

        if text.contains("SSN") || text.contains("Social Security") {
            suggestions.push("Consider redacting Social Security Numbers".to_string());
        }

        if text.contains('@') && text.contains('.') {
            suggestions.push("Consider redacting email addresses".to_string());
        }

        if text.matches(char::is_numeric).count() > 10 {
            suggestions
                .push("Consider redacting phone numbers or other numeric identifiers".to_string());
        }

        Ok(suggestions)
    }

    fn classify_document_type(&self, text: &str) -> Result<String> {
        let text_lower = text.to_lowercase();

        if text_lower.contains("contract") || text_lower.contains("agreement") {
            Ok("Contract".to_string())
        } else if text_lower.contains("medical record") || text_lower.contains("patient") {
            Ok("Medical Record".to_string())
        } else if text_lower.contains("court") || text_lower.contains("filing") {
            Ok("Court Document".to_string())
        } else if text_lower.contains("policy") || text_lower.contains("procedure") {
            Ok("Policy Document".to_string())
        } else {
            Ok("General Document".to_string())
        }
    }

    /// Whether `text` appears to contain sensitive information. `pub(super)`
    /// so [`super::generation`]'s privacy-marker prompt prefix can reuse the
    /// same heuristic rather than duplicating it.
    pub(super) fn contains_sensitive_info(&self, text: &str) -> Result<bool> {
        let text_lower = text.to_lowercase();

        // Check for common sensitive patterns
        let sensitive_patterns = [
            "ssn",
            "social security",
            "dob",
            "date of birth",
            "patient id",
            "medical record",
            "confidential",
        ];

        for pattern in &sensitive_patterns {
            if text_lower.contains(pattern) {
                return Ok(true);
            }
        }

        // Check for email patterns
        if text.contains('@') && text.contains('.') {
            return Ok(true);
        }

        Ok(false)
    }

    fn extract_legal_case_citations(&self, text: &str) -> Result<Vec<Citation>> {
        let mut citations = Vec::new();

        // Look for "v." pattern (Case vs. Case)
        for line in text.lines() {
            if line.contains(" v. ") {
                citations.push(Citation {
                    citation_type: CitationType::LegalCase,
                    text: line.to_string(),
                    page_number: None,
                    authority: None,
                });
            }
        }

        Ok(citations)
    }

    fn extract_statute_citations(&self, text: &str) -> Result<Vec<Citation>> {
        let mut citations = Vec::new();

        // Look for section symbols and USC references
        for line in text.lines() {
            if line.contains('§') || line.contains("U.S.C.") {
                citations.push(Citation {
                    citation_type: CitationType::Statute,
                    text: line.to_string(),
                    page_number: None,
                    authority: Some("U.S. Code".to_string()),
                });
            }
        }

        Ok(citations)
    }

    fn extract_medical_journal_citations(&self, text: &str) -> Result<Vec<Citation>> {
        let mut citations = Vec::new();

        // Look for "et al." pattern
        for line in text.lines() {
            if line.contains("et al.") {
                citations.push(Citation {
                    citation_type: CitationType::MedicalJournal,
                    text: line.to_string(),
                    page_number: None,
                    authority: None,
                });
            }
        }

        Ok(citations)
    }

    fn extract_clinical_trial_citations(&self, text: &str) -> Result<Vec<Citation>> {
        let mut citations = Vec::new();

        // Look for clinical trial identifiers
        for line in text.lines() {
            if line.contains("NCT") || line.contains("clinical trial") {
                citations.push(Citation {
                    citation_type: CitationType::ClinicalTrial,
                    text: line.to_string(),
                    page_number: None,
                    authority: None,
                });
            }
        }

        Ok(citations)
    }

    fn is_medical_domain(&self) -> bool {
        matches!(
            self.config.domain,
            LegalMedicalDomain::Medical
                | LegalMedicalDomain::MedicalClinical
                | LegalMedicalDomain::MedicalResearch
                | LegalMedicalDomain::MedicalPharmacology
                | LegalMedicalDomain::MedicalRadiology
                | LegalMedicalDomain::MedicalPublicHealth
        )
    }

    fn is_legal_domain(&self) -> bool {
        matches!(
            self.config.domain,
            LegalMedicalDomain::Legal
                | LegalMedicalDomain::LegalContract
                | LegalMedicalDomain::LegalLitigation
                | LegalMedicalDomain::LegalRegulatory
                | LegalMedicalDomain::LegalIP
                | LegalMedicalDomain::LegalCriminal
        )
    }

    fn check_hipaa_compliance(&self, text: &str) -> Result<Vec<ComplianceViolation>> {
        let mut violations = Vec::new();

        if self.contains_sensitive_info(text)? {
            violations.push(ComplianceViolation {
                violation_type: "HIPAA Privacy".to_string(),
                severity: "High".to_string(),
                location: "Throughout document".to_string(),
                description: "Document contains potentially sensitive health information"
                    .to_string(),
                suggested_fix: "Apply appropriate redaction or de-identification".to_string(),
            });
        }

        Ok(violations)
    }

    fn check_gdpr_compliance(&self, text: &str) -> Result<Vec<ComplianceViolation>> {
        let mut violations = Vec::new();

        if text.contains('@') && text.contains('.') {
            violations.push(ComplianceViolation {
                violation_type: "GDPR Privacy".to_string(),
                severity: "Medium".to_string(),
                location: "Email addresses".to_string(),
                description: "Document contains email addresses which may be personal data"
                    .to_string(),
                suggested_fix: "Redact or anonymize email addresses".to_string(),
            });
        }

        Ok(violations)
    }

    fn check_attorney_client_privilege(&self, text: &str) -> Result<Vec<ComplianceViolation>> {
        let mut violations = Vec::new();

        if text.to_lowercase().contains("confidential")
            && !text.to_lowercase().contains("privilege")
        {
            violations.push(ComplianceViolation {
                violation_type: "Attorney-Client Privilege".to_string(),
                severity: "High".to_string(),
                location: "Confidential sections".to_string(),
                description: "Document marked confidential but privilege not explicitly claimed"
                    .to_string(),
                suggested_fix: "Add explicit attorney-client privilege statement".to_string(),
            });
        }

        Ok(violations)
    }

    fn generate_compliance_recommendations(
        &self,
        violations: &[ComplianceViolation],
    ) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        for violation in violations {
            recommendations.push(format!(
                "Address {}: {}",
                violation.violation_type, violation.suggested_fix
            ));
        }

        if violations.is_empty() {
            recommendations.push(
                "Document appears to be compliant with basic privacy regulations".to_string(),
            );
        }

        Ok(recommendations)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::tiny_config;
    use super::*;

    fn model() -> LegalMedicalForCausalLM {
        LegalMedicalForCausalLM::new(tiny_config()).expect("model construction")
    }

    #[test]
    fn analyze_document_classifies_a_medical_record() {
        let m = model();
        let analysis = m
            .analyze_document("Patient presented with symptoms consistent with clinical diagnosis")
            .expect("analyze");
        assert_eq!(
            analysis.domain_classification,
            LegalMedicalDomain::MedicalClinical
        );
    }

    #[test]
    fn extract_citations_finds_case_law() {
        let m = model();
        let citations =
            m.extract_citations("Roe v. Wade established a precedent").expect("citations");
        assert!(citations.iter().any(|c| matches!(c.citation_type, CitationType::LegalCase)));
    }

    #[test]
    fn compliance_check_flags_unredacted_confidential_text() {
        let m = model();
        let report = m
            .compliance_check("This document is confidential and contains patient data")
            .expect("compliance");
        assert!(!report.violations.is_empty());
    }
}
