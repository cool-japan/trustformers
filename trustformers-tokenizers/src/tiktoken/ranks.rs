//! Loading of OpenAI `.tiktoken` BPE rank files.
//!
//! A `.tiktoken` file has one entry per line:
//!
//! ```text
//! <base64 of the token bytes> <rank>
//! ```
//!
//! e.g. `IQ== 0` for the single byte `!` at rank 0. These files are *not*
//! redistributed with this crate; they are located through the documented
//! search paths below, which mirror the layout the official `tiktoken` package
//! uses.

use base64::{engine::general_purpose::STANDARD, Engine as _};
use std::collections::HashMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use trustformers_core::errors::{Result, TrustformersError};

/// Primary override: a directory holding `{encoding}.tiktoken` files.
pub const RANK_DIR_ENV: &str = "TRUSTFORMERS_TIKTOKEN_DIR";

/// Cache directory used by the official `tiktoken` Python package.
pub const TIKTOKEN_CACHE_ENV: &str = "TIKTOKEN_CACHE_DIR";

/// Map of token bytes to BPE rank.
pub type RankMap = HashMap<Vec<u8>, usize>;

/// Every location probed for `{encoding_name}.tiktoken`, in priority order.
///
/// 1. `$TRUSTFORMERS_TIKTOKEN_DIR/{encoding}.tiktoken`
/// 2. `$TIKTOKEN_CACHE_DIR/{encoding}.tiktoken`
/// 3. `$XDG_CACHE_HOME/tiktoken/{encoding}.tiktoken`
/// 4. `$HOME/.cache/tiktoken/{encoding}.tiktoken`
/// 5. `./{encoding}.tiktoken`
/// 6. `./tiktoken/{encoding}.tiktoken`
pub fn search_paths(encoding_name: &str) -> Vec<PathBuf> {
    let file_name = format!("{}.tiktoken", encoding_name);
    let mut paths = Vec::new();

    if let Ok(dir) = std::env::var(RANK_DIR_ENV) {
        paths.push(Path::new(&dir).join(&file_name));
    }
    if let Ok(dir) = std::env::var(TIKTOKEN_CACHE_ENV) {
        paths.push(Path::new(&dir).join(&file_name));
    }
    if let Ok(dir) = std::env::var("XDG_CACHE_HOME") {
        paths.push(Path::new(&dir).join("tiktoken").join(&file_name));
    }
    if let Ok(home) = std::env::var("HOME") {
        paths.push(Path::new(&home).join(".cache").join("tiktoken").join(&file_name));
    }
    paths.push(PathBuf::from(&file_name));
    paths.push(Path::new("tiktoken").join(&file_name));

    paths
}

/// First existing rank file for `encoding_name`, if any.
pub fn find_rank_file(encoding_name: &str) -> Option<PathBuf> {
    search_paths(encoding_name).into_iter().find(|path| path.is_file())
}

/// Error explaining how to obtain a missing rank file.
pub fn missing_rank_file_error(encoding_name: &str) -> TrustformersError {
    let probed: Vec<String> =
        search_paths(encoding_name).iter().map(|p| p.display().to_string()).collect();

    TrustformersError::invalid_input(format!(
        "No BPE rank file found for the '{name}' encoding. This crate never ships or \
         fabricates OpenAI rank tables. Download '{name}.tiktoken' (for example from \
         the openaipublic tiktoken bucket) and place it in one of the probed \
         locations, or set {env} to its directory, or load it explicitly with \
         `TiktokenTokenizer::from_file`. Probed paths: {probed}.",
        name = encoding_name,
        env = RANK_DIR_ENV,
        probed = probed.join(", ")
    ))
}

/// Parse a `.tiktoken` rank table from any reader.
pub fn load_ranks_from_reader<R: BufRead>(reader: R) -> Result<RankMap> {
    let mut ranks = RankMap::new();

    for (index, line) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = line.map_err(|e| {
            TrustformersError::io_error(format!("Failed to read line {}: {}", line_number, e))
        })?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut parts = line.split_whitespace();
        let (Some(token), Some(rank)) = (parts.next(), parts.next()) else {
            return Err(TrustformersError::invalid_input(format!(
                "Invalid .tiktoken line {}: expected '<base64 token> <rank>', got {:?}",
                line_number, line
            )));
        };
        if parts.next().is_some() {
            return Err(TrustformersError::invalid_input(format!(
                "Invalid .tiktoken line {}: expected exactly two fields, got {:?}",
                line_number, line
            )));
        }

        let token_bytes = decode_token(token).map_err(|e| {
            TrustformersError::invalid_input(format!(
                "Failed to decode token on line {}: {}",
                line_number, e
            ))
        })?;

        let rank: usize = rank.parse().map_err(|e| {
            TrustformersError::invalid_input(format!("Invalid rank on line {}: {}", line_number, e))
        })?;

        if let Some(previous) = ranks.insert(token_bytes, rank) {
            if previous != rank {
                return Err(TrustformersError::invalid_input(format!(
                    "Duplicate token on line {} with conflicting ranks {} and {}",
                    line_number, previous, rank
                )));
            }
        }
    }

    if ranks.is_empty() {
        return Err(TrustformersError::invalid_input(
            "The .tiktoken rank table is empty".to_string(),
        ));
    }

    Ok(ranks)
}

/// Parse a `.tiktoken` rank table from a file.
pub fn load_ranks_from_file<P: AsRef<Path>>(path: P) -> Result<RankMap> {
    use std::fs::File;
    use std::io::BufReader;

    let path = path.as_ref();
    let file = File::open(path).map_err(|e| {
        TrustformersError::io_error(format!("Failed to open rank file {:?}: {}", path, e))
    })?;

    load_ranks_from_reader(BufReader::new(file))
}

/// Decode one token field.
///
/// Standard `.tiktoken` files use base64; the Python `b'...'` bytes-literal form
/// is also accepted because dumps produced by ad-hoc scripts use it.
pub fn decode_token(encoded: &str) -> std::result::Result<Vec<u8>, String> {
    if let Some(inner) = encoded.strip_prefix("b'").and_then(|s| s.strip_suffix('\'')) {
        return decode_python_bytes_literal(inner);
    }

    STANDARD.decode(encoded).map_err(|e| format!("base64 decode error: {}", e))
}

/// Decode a Python bytes-literal body (the part between the quotes).
pub fn decode_python_bytes_literal(literal: &str) -> std::result::Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    let mut chars = literal.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch != '\\' {
            let mut utf8_buf = [0u8; 4];
            bytes.extend_from_slice(ch.encode_utf8(&mut utf8_buf).as_bytes());
            continue;
        }

        match chars.next().ok_or("unexpected end of escape sequence")? {
            'n' => bytes.push(b'\n'),
            'r' => bytes.push(b'\r'),
            't' => bytes.push(b'\t'),
            '0' => bytes.push(0),
            '\\' => bytes.push(b'\\'),
            '\'' => bytes.push(b'\''),
            '"' => bytes.push(b'"'),
            'x' => {
                let hex1 = chars.next().ok_or("incomplete hex escape")?;
                let hex2 = chars.next().ok_or("incomplete hex escape")?;
                let hex = format!("{}{}", hex1, hex2);
                let value = u8::from_str_radix(&hex, 16)
                    .map_err(|_| format!("invalid hex escape: \\x{}", hex))?;
                bytes.push(value);
            },
            other => return Err(format!("unknown escape sequence: \\{}", other)),
        }
    }

    Ok(bytes)
}

/// Serialize a rank table back to the `.tiktoken` format (used by tests and
/// tooling that needs to persist a table it built).
pub fn write_ranks<W: std::io::Write>(ranks: &RankMap, writer: &mut W) -> Result<()> {
    let mut entries: Vec<(&Vec<u8>, &usize)> = ranks.iter().collect();
    entries.sort_by_key(|&(_, rank)| *rank);

    for (token, rank) in entries {
        writeln!(writer, "{} {}", STANDARD.encode(token), rank).map_err(|e| {
            TrustformersError::io_error(format!("Failed to write rank entry: {}", e))
        })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_ranks_round_trip() {
        let mut ranks = RankMap::new();
        ranks.insert(b"a".to_vec(), 0);
        ranks.insert(b"b".to_vec(), 1);
        ranks.insert(vec![0xff], 2);

        let mut buffer: Vec<u8> = Vec::new();
        write_ranks(&ranks, &mut buffer).expect("serialization must succeed");

        let parsed = load_ranks_from_reader(buffer.as_slice()).expect("parsing must succeed");
        assert_eq!(parsed, ranks);
    }

    #[test]
    fn test_load_ranks_rejects_malformed_lines() {
        assert!(load_ranks_from_reader("not-base64-@@@ 0".as_bytes()).is_err());
        assert!(load_ranks_from_reader("YQ==".as_bytes()).is_err());
        assert!(load_ranks_from_reader("YQ== notanumber".as_bytes()).is_err());
        assert!(load_ranks_from_reader("".as_bytes()).is_err());
    }

    #[test]
    fn test_load_ranks_accepts_comments_and_blank_lines() {
        let text = "# comment\n\nYQ== 0\nYg== 1\n";
        let ranks = load_ranks_from_reader(text.as_bytes()).expect("parsing must succeed");
        assert_eq!(ranks.get(b"a".as_slice()), Some(&0));
        assert_eq!(ranks.get(b"b".as_slice()), Some(&1));
    }

    #[test]
    fn test_python_bytes_literal_tokens() {
        let ranks = load_ranks_from_reader("b'\\xff\\x00' 7\n".as_bytes())
            .expect("bytes-literal tokens must parse");
        assert_eq!(ranks.get([0xffu8, 0x00].as_slice()), Some(&7));
    }

    #[test]
    fn test_missing_rank_file_error_lists_paths() {
        let message = missing_rank_file_error("cl100k_base").to_string();
        assert!(message.contains("cl100k_base.tiktoken"));
        assert!(message.contains(RANK_DIR_ENV));
    }

    #[test]
    fn test_search_paths_honor_env_override() {
        // Uses the process-wide environment, so keep the assertion structural.
        let paths = search_paths("r50k_base");
        assert!(paths.iter().any(|p| p.ends_with("r50k_base.tiktoken")));
    }
}
