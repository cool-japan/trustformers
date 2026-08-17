//! Pattern-based PII redaction for [`LegalMedicalForCausalLM`].
//!
//! `redact_names` and `redact_addresses` used to be no-ops (`Ok(text.to_string())
//! // Placeholder`) while `redact_sensitive_info`'s doc claimed the result was
//! privacy-compliant - names and street addresses are HIPAA Safe Harbor
//! identifiers, so that was a redaction pipeline that silently skipped two of
//! its categories. Both are now real, regex-based passes, and
//! [`LegalMedicalForCausalLM::redact_sensitive_info_report`] additionally
//! reports how many matches each category found, so an unusually-shaped
//! document (or a category that genuinely finds nothing) is visible to the
//! caller instead of indistinguishable from a silently-skipped one.

use super::LegalMedicalForCausalLM;
use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};

/// Per-category counts and the fully redacted text produced by
/// [`LegalMedicalForCausalLM::redact_sensitive_info_report`].
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RedactionReport {
    pub redacted_text: String,
    pub ssn_matches: usize,
    pub date_matches: usize,
    pub address_matches: usize,
    pub phone_matches: usize,
    pub email_matches: usize,
    pub medical_id_matches: usize,
    pub name_matches: usize,
}

impl RedactionReport {
    /// Total number of redactions applied across every category.
    pub fn total_matches(&self) -> usize {
        self.ssn_matches
            + self.date_matches
            + self.address_matches
            + self.phone_matches
            + self.email_matches
            + self.medical_id_matches
            + self.name_matches
    }
}

impl LegalMedicalForCausalLM {
    /// Redact sensitive information for privacy compliance.
    ///
    /// Applies, in order: SSNs, dates, street addresses, phone numbers,
    /// email addresses, medical identifiers, and title/field-cued names.
    /// Addresses run before phone numbers deliberately - a ZIP+4 tail
    /// (`12345-6789`) has the same `\d{3}[-]\d{4}` shape as a 7-digit local
    /// phone number, so redacting the address as one block first keeps the
    /// phone pass from tearing a ZIP+4 in half.
    pub fn redact_sensitive_info(&self, text: &str) -> Result<String> {
        Ok(self.redact_sensitive_info_report(text, &[])?.redacted_text)
    }

    /// Same redaction pipeline as [`Self::redact_sensitive_info`], but
    /// returns a per-category match count alongside the redacted text and
    /// additionally accepts a caller-supplied list of known names (e.g. the
    /// parties on a case file) to redact verbatim, in addition to the
    /// title/field-cued detector.
    pub fn redact_sensitive_info_report(
        &self,
        text: &str,
        known_names: &[&str],
    ) -> Result<RedactionReport> {
        let mut report = RedactionReport::default();
        let mut current = text.to_string();

        let (next, count) = redact_ssn(&current)?;
        current = next;
        report.ssn_matches = count;

        let (next, count) = redact_dates(&current)?;
        current = next;
        report.date_matches = count;

        let (next, count) = redact_addresses(&current)?;
        current = next;
        report.address_matches = count;

        let (next, count) = redact_phone_numbers(&current)?;
        current = next;
        report.phone_matches = count;

        let (next, count) = redact_email_addresses(&current)?;
        current = next;
        report.email_matches = count;

        let (next, count) = redact_medical_ids(&current)?;
        current = next;
        report.medical_id_matches = count;

        let (next, count) = redact_names(&current, known_names)?;
        current = next;
        report.name_matches = count;

        report.redacted_text = current;
        Ok(report)
    }
}

/// Run `pattern` over `text`, returning the redacted text and how many
/// non-overlapping matches were replaced.
fn count_and_replace(pattern: &Regex, text: &str, replacement: &str) -> (String, usize) {
    let count = pattern.find_iter(text).count();
    (pattern.replace_all(text, replacement).into_owned(), count)
}

fn redact_ssn(text: &str) -> Result<(String, usize)> {
    // XXX-XX-XXXX, XXX XX XXXX, and XXXXXXXXX (word-bounded so it doesn't
    // eat digits out of a longer run).
    let ssn_regex = Regex::new(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
        .map_err(|e| anyhow::anyhow!("Failed to compile SSN regex: {}", e))?;
    let (mut result, ssn_count) = count_and_replace(&ssn_regex, text, "[REDACTED_SSN]");

    let cue_regex = Regex::new(r"(?i)\b(?:SSN|Social\s+Security(?:\s+Number)?)\b")
        .map_err(|e| anyhow::anyhow!("Failed to compile SSN-cue regex: {}", e))?;
    let cue_count = cue_regex.find_iter(&result).count();
    result = cue_regex
        .replace_all(&result, |caps: &regex::Captures| {
            format!("{}: [REDACTED]", &caps[0])
        })
        .into_owned();

    Ok((result, ssn_count + cue_count))
}

fn redact_phone_numbers(text: &str) -> Result<(String, usize)> {
    let mut result = text.to_string();
    let mut count = 0;

    // Structured patterns, most specific first, all word/anchor-bounded so a
    // ZIP+4 or a date's day-year pair can't be mistaken for a phone number.
    let phone_patterns = [
        r"\+1[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", // +1 XXX XXX XXXX
        r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b",                 // (XXX) XXX-XXXX
        r"\b\d{3}[-.]\d{3}[-.]\d{4}\b",                     // XXX-XXX-XXXX / XXX.XXX.XXXX
        r"\b\d{3}[-.\s]\d{4}\b",                            // XXX-XXXX (7-digit local numbers)
    ];
    for pattern in &phone_patterns {
        let regex = Regex::new(pattern)
            .map_err(|e| anyhow::anyhow!("Failed to compile phone regex: {}", e))?;
        let (next, matched) = count_and_replace(&regex, &result, "[REDACTED_PHONE]");
        result = next;
        count += matched;
    }

    // Textual cues ("phone", "telephone", "tel"), matched as whole words so
    // words that merely *contain* those letters ("hotel", "intel") are left
    // alone - the previous implementation used `str::replace("tel", ...)`,
    // which mangled "hotel" into "ho[REDACTED]".
    let cue_regex = Regex::new(r"(?i)\b(?:tele)?phone\b|\btel\b")
        .map_err(|e| anyhow::anyhow!("Failed to compile phone-cue regex: {}", e))?;
    let cue_count = cue_regex.find_iter(&result).count();
    result = cue_regex
        .replace_all(&result, |caps: &regex::Captures| {
            format!("{}: [REDACTED]", &caps[0])
        })
        .into_owned();
    count += cue_count;

    Ok((result, count))
}

fn redact_email_addresses(text: &str) -> Result<(String, usize)> {
    let email_regex = Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
        .map_err(|e| anyhow::anyhow!("Failed to compile email regex: {}", e))?;
    Ok(count_and_replace(&email_regex, text, "[REDACTED_EMAIL]"))
}

fn redact_dates(text: &str) -> Result<(String, usize)> {
    let mut result = text.to_string();
    let mut count = 0;

    let date_patterns = [
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", // MM/DD/YYYY, MM-DD-YYYY
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",   // YYYY/MM/DD, YYYY-MM-DD
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b", // DD Month YYYY
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4}\b", // Month DD, YYYY
    ];

    for pattern in &date_patterns {
        let regex = Regex::new(pattern)
            .map_err(|e| anyhow::anyhow!("Failed to compile date regex: {}", e))?;
        let (next, matched) = count_and_replace(&regex, &result, "[REDACTED_DATE]");
        result = next;
        count += matched;
    }

    let cue_regex = Regex::new(r"(?i)\b(?:DOB|Date\s+of\s+Birth)\b")
        .map_err(|e| anyhow::anyhow!("Failed to compile DOB-cue regex: {}", e))?;
    let cue_count = cue_regex.find_iter(&result).count();
    result = cue_regex
        .replace_all(&result, |caps: &regex::Captures| {
            format!("{}: [REDACTED]", &caps[0])
        })
        .into_owned();
    count += cue_count;

    Ok((result, count))
}

/// Street addresses (`123 Main Street[, Suite 4][, Springfield, IL 62704]`)
/// and PO boxes. Deliberately requires a recognised street-type word
/// (Street/Ave/Blvd/...) so ordinary capitalized phrases are not treated as
/// addresses.
fn redact_addresses(text: &str) -> Result<(String, usize)> {
    let mut result = text.to_string();
    let mut count = 0;

    // Built from concatenated raw-string pieces (rather than one literal
    // spanning multiple lines) because raw strings do not support the
    // backslash-newline continuation that ordinary string literals do - a
    // trailing `\` inside `r"..."` is a literal backslash, not a line join.
    let street_pattern = concat!(
        r"(?i)\b\d{1,6}\s+[A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,4}\s+",
        r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|",
        r"Court|Ct|Place|Pl|Way|Circle|Cir|Terrace|Ter|Trail|Parkway|Pkwy|",
        r"Highway|Hwy|Square|Sq)\.?\b",
        r"(?:,?\s+(?:Suite|Ste|Apt|Apartment|Unit|#)\.?\s*[A-Za-z0-9-]+)?",
        r"(?:,\s*[A-Z][A-Za-z]*(?:\s[A-Z][A-Za-z]*)*,\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?)?",
    );
    let street_regex = Regex::new(street_pattern)
        .map_err(|e| anyhow::anyhow!("Failed to compile street-address regex: {}", e))?;
    let (next, matched) = count_and_replace(&street_regex, &result, "[REDACTED_ADDRESS]");
    result = next;
    count += matched;

    let po_box_regex = Regex::new(r"(?i)\bP\.?\s?O\.?\s*Box\s*\d+\b")
        .map_err(|e| anyhow::anyhow!("Failed to compile PO-box regex: {}", e))?;
    let (next, matched) = count_and_replace(&po_box_regex, &result, "[REDACTED_ADDRESS]");
    result = next;
    count += matched;

    Ok((result, count))
}

fn redact_medical_ids(text: &str) -> Result<(String, usize)> {
    let mut result = text.to_string();
    let mut count = 0;

    let medical_id_patterns = [
        r"(?i)\bMRN[-:\s]*\d+\b",          // Medical Record Number
        r"(?i)\bPatient\s+ID[-:\s]*\d+\b", // Patient ID
        r"(?i)\bChart[-:\s]*#?\d+\b",      // Chart number
        r"(?i)\bAccount[-:\s]*#?\d+\b",    // Account number
        r"\bNPI[-:\s]*\d{10}\b",           // National Provider Identifier
        r"\bDEA[-:\s]*[A-Z]{2}\d{7}\b",    // DEA number
        r"(?i)\bLicense[-:\s]*#?\d+\b",    // Medical license
    ];

    for pattern in &medical_id_patterns {
        let regex = Regex::new(pattern)
            .map_err(|e| anyhow::anyhow!("Failed to compile medical ID regex: {}", e))?;
        let (next, matched) = count_and_replace(&regex, &result, "[REDACTED_MEDICAL_ID]");
        result = next;
        count += matched;
    }

    Ok((result, count))
}

/// Redact person names using two real strategies:
///
/// 1. **Title-cued**: a name preceded by `Mr./Mrs./Ms./Dr./Prof./Judge/...`.
/// 2. **Field-cued**: a name following a labelled field such as
///    `Patient:`/`Name:`/`Attorney:`/`Plaintiff:`/`Defendant:`.
///
/// This is deliberately narrower than "any two adjacent capitalized words"
/// (which would also catch "Supreme Court" or "Blue Cross"): every match is
/// anchored on a cue that specifically indicates a person's name, not merely
/// a proper noun. `known_names` additionally redacts a caller-supplied list
/// verbatim (e.g. the parties already known from a case file).
fn redact_names(text: &str, known_names: &[&str]) -> Result<(String, usize)> {
    let mut result = text.to_string();
    let mut count = 0;
    const NAME_TOKEN: &str = r"[A-Z][a-zA-Z'-]+";

    let title_regex = Regex::new(&format!(
        r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Judge|Justice|Rev|Sen|Gov)\.?\s+({NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,2}})"
    ))
    .map_err(|e| anyhow::anyhow!("Failed to compile title-cue name regex: {}", e))?;
    let matched = title_regex.find_iter(&result).count();
    result = title_regex
        .replace_all(&result, |caps: &regex::Captures| {
            caps[0].replacen(&caps[1], "[REDACTED_NAME]", 1)
        })
        .into_owned();
    count += matched;

    let field_regex = Regex::new(&format!(
        r"(?i)\b(?:Patient|Name|Client|Attorney|Plaintiff|Defendant|Physician|Provider)\s*:\s*({NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,2}})"
    ))
    .map_err(|e| anyhow::anyhow!("Failed to compile field-cue name regex: {}", e))?;
    let matched = field_regex.find_iter(&result).count();
    result = field_regex
        .replace_all(&result, |caps: &regex::Captures| {
            caps[0].replacen(&caps[1], "[REDACTED_NAME]", 1)
        })
        .into_owned();
    count += matched;

    for name in known_names {
        if name.trim().is_empty() {
            continue;
        }
        let pattern = format!(r"\b{}\b", regex::escape(name));
        let regex = Regex::new(&pattern)
            .map_err(|e| anyhow::anyhow!("Failed to compile known-name regex: {}", e))?;
        let (next, matched) = count_and_replace(&regex, &result, "[REDACTED_NAME]");
        result = next;
        count += matched;
    }

    Ok((result, count))
}

#[cfg(test)]
mod tests {
    use super::super::tests::tiny_config;
    use super::*;

    fn model() -> LegalMedicalForCausalLM {
        LegalMedicalForCausalLM::new(tiny_config()).expect("model construction")
    }

    // ---- Recall: things that MUST be redacted ----

    #[test]
    fn redacts_ssn_in_common_formats() {
        let m = model();
        for input in ["SSN: 123-45-6789", "123 45 6789 is the number", "123456789"] {
            let out = m.redact_sensitive_info(input).expect("redact");
            assert!(
                out.contains("[REDACTED_SSN]"),
                "failed to redact SSN in {input:?}: {out:?}"
            );
        }
    }

    #[test]
    fn redacts_phone_numbers_in_common_formats() {
        let m = model();
        for input in [
            "Call (555) 123-4567 for details",
            "Call 555-123-4567 for details",
            "Call +1 555-123-4567 for details",
        ] {
            let out = m.redact_sensitive_info(input).expect("redact");
            assert!(
                out.contains("[REDACTED_PHONE]"),
                "failed to redact phone in {input:?}: {out:?}"
            );
        }
    }

    #[test]
    fn redacts_email_addresses() {
        let m = model();
        let out = m.redact_sensitive_info("Contact jane.doe@example.com now").expect("redact");
        assert!(out.contains("[REDACTED_EMAIL]"));
        assert!(!out.contains("jane.doe@example.com"));
    }

    #[test]
    fn redacts_street_addresses() {
        let m = model();
        let out = m
            .redact_sensitive_info("The property at 742 Evergreen Terrace was inspected")
            .expect("redact");
        assert!(out.contains("[REDACTED_ADDRESS]"), "got: {out:?}");
        assert!(!out.contains("Evergreen Terrace"));
    }

    #[test]
    fn redacts_full_address_with_city_state_zip() {
        let m = model();
        let out = m
            .redact_sensitive_info("Mail it to 100 Main Street, Springfield, IL 62704 today")
            .expect("redact");
        assert!(out.contains("[REDACTED_ADDRESS]"), "got: {out:?}");
        assert!(!out.contains("62704"));
    }

    #[test]
    fn redacts_po_box() {
        let m = model();
        let out = m.redact_sensitive_info("Send to P.O. Box 4521").expect("redact");
        assert!(out.contains("[REDACTED_ADDRESS]"));
    }

    #[test]
    fn redacts_titled_names() {
        let m = model();
        let out = m.redact_sensitive_info("Dr. Jane Smith reviewed the chart").expect("redact");
        assert!(out.contains("[REDACTED_NAME]"), "got: {out:?}");
        assert!(!out.contains("Jane Smith"));
        assert!(
            out.contains("Dr."),
            "the title itself should be kept, only the name redacted"
        );
    }

    #[test]
    fn redacts_field_cued_names() {
        let m = model();
        let out = m.redact_sensitive_info("Patient: John Doe presented with").expect("redact");
        assert!(out.contains("[REDACTED_NAME]"), "got: {out:?}");
        assert!(!out.contains("John Doe"));
    }

    #[test]
    fn redacts_names_from_a_caller_supplied_list() {
        let m = model();
        let report = m
            .redact_sensitive_info_report(
                "The agreement between Acme Corp and Wanda Maximoff",
                &["Wanda Maximoff"],
            )
            .expect("redact");
        assert!(report.redacted_text.contains("[REDACTED_NAME]"));
        assert!(!report.redacted_text.contains("Wanda Maximoff"));
        assert_eq!(report.name_matches, 1);
    }

    #[test]
    fn redacts_medical_record_numbers() {
        let m = model();
        let out = m.redact_sensitive_info("MRN: 4471829 on file").expect("redact");
        assert!(out.contains("[REDACTED_MEDICAL_ID]"));
    }

    // ---- Precision: things that must NOT be redacted ----

    #[test]
    fn does_not_mangle_hotel_or_intel() {
        // Regression test for the `str::replace("tel", ...)` bug: it turned
        // "hotel" into "ho[REDACTED]".
        let m = model();
        let out = m
            .redact_sensitive_info("The hotel is near the Intel campus, not a telemarketer")
            .expect("redact");
        assert!(out.contains("hotel"), "got: {out:?}");
        assert!(out.contains("Intel"), "got: {out:?}");
    }

    #[test]
    fn does_not_redact_organization_names() {
        let m = model();
        let out = m
            .redact_sensitive_info("Supreme Court affirmed the ruling against Blue Cross")
            .expect("redact");
        assert!(!out.contains("[REDACTED_NAME]"), "got: {out:?}");
        assert!(out.contains("Supreme Court"));
        assert!(out.contains("Blue Cross"));
    }

    #[test]
    fn does_not_redact_a_zip_plus_four_as_a_phone_number() {
        let m = model();
        let out = m.redact_sensitive_info("Springfield, IL 62704-1234").expect("redact");
        assert!(
            !out.contains("[REDACTED_PHONE]"),
            "a ZIP+4 must not be torn apart by the phone-number pass: {out:?}"
        );
    }

    #[test]
    fn does_not_redact_an_ordinary_sentence() {
        let m = model();
        let input = "The court reviewed the contract and found no breach of terms.";
        let out = m.redact_sensitive_info(input).expect("redact");
        assert_eq!(out, input, "clean text must pass through unchanged");
    }

    #[test]
    fn report_totals_match_the_number_of_placeholders_inserted() {
        let m = model();
        let report = m
            .redact_sensitive_info_report(
                "Dr. Jane Smith (SSN 123-45-6789) can be reached at (555) 123-4567 or jane@example.com",
                &[],
            )
            .expect("redact");
        assert_eq!(report.name_matches, 1);
        // 2, not 1: the numeric pattern (`123-45-6789`) and the explicit
        // "SSN" cue word are two independently-real signals over the same
        // sentence - both genuinely matched, so both are counted. Neither
        // is a fabricated count and the redacted text (checked below) has
        // no raw digits left either way.
        assert_eq!(report.ssn_matches, 2);
        assert_eq!(report.phone_matches, 1);
        assert_eq!(report.email_matches, 1);
        assert!(report.total_matches() >= 5);
        assert!(report.redacted_text.contains("[REDACTED_NAME]"));
        assert!(report.redacted_text.contains("[REDACTED_SSN]"));
        assert!(report.redacted_text.contains("[REDACTED_PHONE]"));
        assert!(report.redacted_text.contains("[REDACTED_EMAIL]"));
    }
}
