//! A minimal, dependency-free XML reader.
//!
//! Just enough of XML 1.0 to read machine-generated documents such as
//! Valgrind's `--xml=yes` memcheck output: elements, attributes, character
//! data, comments, processing instructions, CDATA sections and the five
//! predefined entities. Namespaces, DTDs and external entities are not
//! supported — a document that uses them is rejected rather than
//! mis-interpreted.

use std::collections::HashMap;
use std::fmt;

/// Error produced while reading an XML document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XmlError {
    /// Human readable description of the problem.
    pub message: String,
    /// Byte offset in the input where the problem was detected.
    pub offset: usize,
}

impl fmt::Display for XmlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "XML parse error at byte {}: {}",
            self.offset, self.message
        )
    }
}

impl std::error::Error for XmlError {}

/// One XML element.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct XmlNode {
    /// Element name.
    pub name: String,
    /// Element attributes.
    pub attributes: HashMap<String, String>,
    /// Concatenated character data directly inside this element.
    pub text: String,
    /// Child elements, in document order.
    pub children: Vec<XmlNode>,
}

impl XmlNode {
    /// First direct child with the given name.
    pub fn child(&self, name: &str) -> Option<&XmlNode> {
        self.children.iter().find(|child| child.name == name)
    }

    /// All direct children with the given name.
    pub fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a XmlNode> + 'a {
        self.children.iter().filter(move |child| child.name == name)
    }

    /// Trimmed character data of this element.
    pub fn text_trim(&self) -> &str {
        self.text.trim()
    }

    /// Trimmed character data of the first direct child with the given name.
    pub fn child_text(&self, name: &str) -> Option<&str> {
        self.child(name).map(|child| child.text_trim())
    }

    /// Recursively find the first descendant (or self) with the given name.
    pub fn find_descendant(&self, name: &str) -> Option<&XmlNode> {
        if self.name == name {
            return Some(self);
        }
        self.children.iter().find_map(|child| child.find_descendant(name))
    }
}

/// Parse an XML document and return its root element.
pub fn parse(input: &str) -> Result<XmlNode, XmlError> {
    Parser {
        bytes: input.as_bytes(),
        position: 0,
        input,
    }
    .parse_document()
}

struct Parser<'a> {
    bytes: &'a [u8],
    position: usize,
    input: &'a str,
}

impl<'a> Parser<'a> {
    fn error<T>(&self, message: impl Into<String>) -> Result<T, XmlError> {
        Err(XmlError {
            message: message.into(),
            offset: self.position,
        })
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn starts_with(&self, prefix: &str) -> bool {
        self.input[self.position..].starts_with(prefix)
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.position += 1;
        }
    }

    /// Skip comments, processing instructions and DOCTYPE-less prologue noise.
    fn skip_misc(&mut self) -> Result<(), XmlError> {
        loop {
            self.skip_whitespace();
            if self.starts_with("<!--") {
                let end = self.input[self.position..].find("-->").ok_or_else(|| XmlError {
                    message: "unterminated comment".to_string(),
                    offset: self.position,
                })?;
                self.position += end + 3;
            } else if self.starts_with("<?") {
                let end = self.input[self.position..].find("?>").ok_or_else(|| XmlError {
                    message: "unterminated processing instruction".to_string(),
                    offset: self.position,
                })?;
                self.position += end + 2;
            } else if self.starts_with("<!DOCTYPE") {
                return self.error("DOCTYPE declarations are not supported");
            } else {
                return Ok(());
            }
        }
    }

    fn parse_document(mut self) -> Result<XmlNode, XmlError> {
        self.skip_misc()?;
        let root = self.parse_element()?;
        self.skip_misc()?;
        if self.position != self.bytes.len() {
            return self.error("trailing content after the root element");
        }
        Ok(root)
    }

    fn parse_name(&mut self) -> Result<String, XmlError> {
        let start = self.position;
        while let Some(byte) = self.peek() {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':') {
                self.position += 1;
            } else {
                break;
            }
        }
        if start == self.position {
            return self.error("expected a name");
        }
        Ok(self.input[start..self.position].to_string())
    }

    fn parse_attribute_value(&mut self) -> Result<String, XmlError> {
        let quote = match self.peek() {
            Some(q @ (b'"' | b'\'')) => q,
            _ => return self.error("expected a quoted attribute value"),
        };
        self.position += 1;
        let start = self.position;
        while let Some(byte) = self.peek() {
            if byte == quote {
                let raw = &self.input[start..self.position];
                self.position += 1;
                return decode_entities(raw).map_err(|message| XmlError {
                    message,
                    offset: start,
                });
            }
            self.position += 1;
        }
        self.error("unterminated attribute value")
    }

    fn parse_element(&mut self) -> Result<XmlNode, XmlError> {
        if self.peek() != Some(b'<') {
            return self.error("expected '<'");
        }
        self.position += 1;
        let name = self.parse_name()?;

        let mut attributes = HashMap::new();
        loop {
            self.skip_whitespace();
            match self.peek() {
                Some(b'/') => {
                    self.position += 1;
                    if self.peek() != Some(b'>') {
                        return self.error("expected '>' after '/'");
                    }
                    self.position += 1;
                    return Ok(XmlNode {
                        name,
                        attributes,
                        text: String::new(),
                        children: Vec::new(),
                    });
                },
                Some(b'>') => {
                    self.position += 1;
                    break;
                },
                Some(_) => {
                    let attribute_name = self.parse_name()?;
                    self.skip_whitespace();
                    if self.peek() != Some(b'=') {
                        return self.error("expected '=' in attribute");
                    }
                    self.position += 1;
                    self.skip_whitespace();
                    let value = self.parse_attribute_value()?;
                    attributes.insert(attribute_name, value);
                },
                None => return self.error("unterminated start tag"),
            }
        }

        let mut text = String::new();
        let mut children = Vec::new();

        loop {
            if self.position >= self.bytes.len() {
                return self.error(format!("unclosed element <{}>", name));
            }

            if self.starts_with("</") {
                self.position += 2;
                let closing = self.parse_name()?;
                self.skip_whitespace();
                if self.peek() != Some(b'>') {
                    return self.error("expected '>' in end tag");
                }
                self.position += 1;
                if closing != name {
                    return self.error(format!("</{}> does not close <{}>", closing, name));
                }
                return Ok(XmlNode {
                    name,
                    attributes,
                    text,
                    children,
                });
            }

            if self.starts_with("<!--") {
                let end = self.input[self.position..].find("-->").ok_or_else(|| XmlError {
                    message: "unterminated comment".to_string(),
                    offset: self.position,
                })?;
                self.position += end + 3;
                continue;
            }

            if self.starts_with("<![CDATA[") {
                self.position += 9;
                let end = self.input[self.position..].find("]]>").ok_or_else(|| XmlError {
                    message: "unterminated CDATA section".to_string(),
                    offset: self.position,
                })?;
                text.push_str(&self.input[self.position..self.position + end]);
                self.position += end + 3;
                continue;
            }

            if self.starts_with("<?") {
                let end = self.input[self.position..].find("?>").ok_or_else(|| XmlError {
                    message: "unterminated processing instruction".to_string(),
                    offset: self.position,
                })?;
                self.position += end + 2;
                continue;
            }

            if self.peek() == Some(b'<') {
                children.push(self.parse_element()?);
                continue;
            }

            // Character data up to the next '<'.
            let start = self.position;
            while let Some(byte) = self.peek() {
                if byte == b'<' {
                    break;
                }
                self.position += 1;
            }
            let raw = &self.input[start..self.position];
            let decoded = decode_entities(raw).map_err(|message| XmlError {
                message,
                offset: start,
            })?;
            text.push_str(&decoded);
        }
    }
}

/// Expand the five predefined XML entities plus numeric character references.
fn decode_entities(raw: &str) -> Result<String, String> {
    if !raw.contains('&') {
        return Ok(raw.to_string());
    }

    let mut output = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(index) = rest.find('&') {
        output.push_str(&rest[..index]);
        let tail = &rest[index..];
        let end = tail.find(';').ok_or_else(|| "unterminated entity reference".to_string())?;
        let entity = &tail[1..end];
        match entity {
            "amp" => output.push('&'),
            "lt" => output.push('<'),
            "gt" => output.push('>'),
            "quot" => output.push('"'),
            "apos" => output.push('\''),
            numeric if numeric.starts_with('#') => {
                let code = if let Some(hex) = numeric.strip_prefix("#x") {
                    u32::from_str_radix(hex, 16)
                        .map_err(|_| format!("invalid hex character reference &{};", numeric))?
                } else {
                    numeric[1..]
                        .parse::<u32>()
                        .map_err(|_| format!("invalid character reference &{};", numeric))?
                };
                let character = char::from_u32(code).ok_or_else(|| {
                    format!("character reference &{}; is not a scalar value", numeric)
                })?;
                output.push(character);
            },
            other => return Err(format!("unsupported entity reference &{};", other)),
        }
        rest = &tail[end + 1..];
    }
    output.push_str(rest);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_document() {
        let document = parse("<root><a>1</a><a>2</a><b x=\"y\"/></root>").expect("parse failed");
        assert_eq!(document.name, "root");
        assert_eq!(document.children_named("a").count(), 2);
        assert_eq!(document.child_text("a"), Some("1"));
        assert_eq!(
            document.child("b").and_then(|b| b.attributes.get("x")).map(String::as_str),
            Some("y")
        );
    }

    #[test]
    fn test_prologue_and_comments_are_skipped() {
        let document = parse(
            "<?xml version=\"1.0\"?>\n<!-- a comment -->\n<root>\n  <value>7</value>\n</root>\n",
        )
        .expect("parse failed");
        assert_eq!(document.child_text("value"), Some("7"));
    }

    #[test]
    fn test_entities_and_cdata() {
        let document = parse("<root><t>a &amp; b &lt;c&gt;</t><u><![CDATA[<raw> & ]]></u></root>")
            .expect("parse failed");
        assert_eq!(document.child_text("t"), Some("a & b <c>"));
        assert_eq!(document.child_text("u"), Some("<raw> &"));
    }

    #[test]
    fn test_numeric_character_reference() {
        let document = parse("<root>&#65;&#x42;</root>").expect("parse failed");
        assert_eq!(document.text_trim(), "AB");
    }

    #[test]
    fn test_mismatched_tag_is_an_error() {
        let error = parse("<root><a></b></root>").expect_err("must reject mismatched tags");
        assert!(error.message.contains("does not close"));
    }

    #[test]
    fn test_unclosed_element_is_an_error() {
        assert!(parse("<root><a></root>").is_err());
    }

    #[test]
    fn test_nested_lookup() {
        let document = parse("<a><b><c><d>deep</d></c></b></a>").expect("parse failed");
        assert_eq!(
            document.find_descendant("d").map(|node| node.text_trim()),
            Some("deep")
        );
    }
}
