//! A minimal Python pickle interpreter for the opcode subset `torch.save` emits.
//!
//! This is deliberately *not* a general unpickler: it never imports or calls
//! anything. `GLOBAL`/`STACK_GLOBAL` push a symbolic
//! [`PickleValue::Global`], `REDUCE` records the call rather than performing it,
//! and `BINPERSID` records the persistent id. The torch reader then interprets
//! those records — which is both safe (no arbitrary code execution, unlike CPython's
//! `pickle.load`) and sufficient, because a state dict only ever "calls"
//! `torch._utils._rebuild_tensor_v2` and friends.
//!
//! Protocols 2 through 5 are covered. An opcode outside the supported set produces
//! an error naming it, so an unsupported checkpoint fails loudly.

use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// A value on the pickle stack.
#[derive(Debug, Clone, PartialEq)]
pub enum PickleValue {
    /// `None`.
    None,
    /// `True` / `False`.
    Bool(bool),
    /// Any Python integer that fits in `i64`.
    Int(i64),
    /// A Python float.
    Float(f64),
    /// A `bytes` object.
    Bytes(Vec<u8>),
    /// A `str` object.
    String(String),
    /// A `list`.
    List(Vec<PickleValue>),
    /// A `tuple`.
    Tuple(Vec<PickleValue>),
    /// A mapping, preserving insertion order.
    Dict(Vec<(PickleValue, PickleValue)>),
    /// A `set` or `frozenset`.
    Set(Vec<PickleValue>),
    /// A symbolic reference to `module.name`; never imported.
    Global { module: String, name: String },
    /// A recorded `callable(*args)` that this interpreter does not perform.
    Reduce {
        callable: Box<PickleValue>,
        args: Box<PickleValue>,
    },
    /// A recorded object with state applied by `BUILD`.
    Object {
        callable: Box<PickleValue>,
        args: Box<PickleValue>,
        state: Box<PickleValue>,
    },
    /// The argument of `PERSID` / `BINPERSID`.
    PersistentId(Box<PickleValue>),
}

impl PickleValue {
    /// The string payload, if this is a `str`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            PickleValue::String(value) => Some(value),
            _ => None,
        }
    }

    /// The integer payload, if this is an `int`.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            PickleValue::Int(value) => Some(*value),
            PickleValue::Bool(value) => Some(i64::from(*value)),
            _ => None,
        }
    }

    /// The element list, if this is a `tuple` or `list`.
    pub fn as_sequence(&self) -> Option<&[PickleValue]> {
        match self {
            PickleValue::Tuple(items) | PickleValue::List(items) => Some(items),
            _ => None,
        }
    }

    /// The key/value pairs, if this is a mapping.
    pub fn as_dict(&self) -> Option<&[(PickleValue, PickleValue)]> {
        match self {
            PickleValue::Dict(items) => Some(items),
            _ => None,
        }
    }

    /// Look up a string key in a mapping.
    pub fn get(&self, key: &str) -> Option<&PickleValue> {
        self.as_dict()?
            .iter()
            .find(|(k, _)| k.as_str() == Some(key))
            .map(|(_, value)| value)
    }
}

/// Interpret a pickle stream, returning the value left by `STOP`.
pub fn load(bytes: &[u8]) -> Result<PickleValue> {
    Machine::new(bytes).run()
}

struct Machine<'a> {
    bytes: &'a [u8],
    position: usize,
    stack: Vec<PickleValue>,
    /// Saved stacks, one per open `MARK`.
    marks: Vec<Vec<PickleValue>>,
    memo: HashMap<u64, PickleValue>,
}

impl<'a> Machine<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            position: 0,
            stack: Vec::new(),
            marks: Vec::new(),
            memo: HashMap::new(),
        }
    }

    fn run(mut self) -> Result<PickleValue> {
        loop {
            let opcode = self.read_u8()?;
            match opcode {
                b'\x80' => {
                    // PROTO
                    let protocol = self.read_u8()?;
                    if protocol > 5 {
                        return Err(anyhow!(
                            "pickle protocol {protocol} is newer than this reader supports (2-5)"
                        ));
                    }
                },
                b'\x95' => {
                    // FRAME: a length hint we can safely ignore.
                    self.read_u64()?;
                },
                b'.' => {
                    // STOP
                    return self
                        .stack
                        .pop()
                        .ok_or_else(|| anyhow!("pickle: STOP with an empty stack"));
                },

                // --- markers and stack manipulation ---
                b'(' => {
                    self.marks.push(std::mem::take(&mut self.stack));
                },
                b'0' => {
                    self.pop()?;
                },
                b'1' => {
                    self.pop_mark()?;
                },
                b'2' => {
                    let top = self
                        .stack
                        .last()
                        .cloned()
                        .ok_or_else(|| anyhow!("pickle: DUP on empty stack"))?;
                    self.stack.push(top);
                },

                // --- constants ---
                b'N' => self.stack.push(PickleValue::None),
                b'\x88' => self.stack.push(PickleValue::Bool(true)),
                b'\x89' => self.stack.push(PickleValue::Bool(false)),

                // --- integers ---
                b'I' => {
                    let line = self.read_line()?;
                    let value = match line.as_str() {
                        "01" => PickleValue::Bool(true),
                        "00" => PickleValue::Bool(false),
                        other => PickleValue::Int(
                            other
                                .parse::<i64>()
                                .map_err(|e| anyhow!("pickle: bad INT '{other}': {e}"))?,
                        ),
                    };
                    self.stack.push(value);
                },
                b'J' => {
                    let value = self.read_i32()?;
                    self.stack.push(PickleValue::Int(i64::from(value)));
                },
                b'K' => {
                    let value = self.read_u8()?;
                    self.stack.push(PickleValue::Int(i64::from(value)));
                },
                b'M' => {
                    let value = self.read_u16()?;
                    self.stack.push(PickleValue::Int(i64::from(value)));
                },
                b'L' => {
                    let line = self.read_line()?;
                    let trimmed = line.strip_suffix('L').unwrap_or(&line);
                    self.stack.push(PickleValue::Int(
                        trimmed
                            .parse::<i64>()
                            .map_err(|e| anyhow!("pickle: bad LONG '{trimmed}': {e}"))?,
                    ));
                },
                b'\x8a' => {
                    let len = self.read_u8()? as usize;
                    let value = self.read_signed_long(len)?;
                    self.stack.push(PickleValue::Int(value));
                },
                b'\x8b' => {
                    let len = self.read_i32()?;
                    if len < 0 {
                        return Err(anyhow!("pickle: LONG4 with negative length"));
                    }
                    let value = self.read_signed_long(len as usize)?;
                    self.stack.push(PickleValue::Int(value));
                },

                // --- floats ---
                b'G' => {
                    let bits = self.read_be_u64()?;
                    self.stack.push(PickleValue::Float(f64::from_bits(bits)));
                },
                b'F' => {
                    let line = self.read_line()?;
                    self.stack.push(PickleValue::Float(
                        line.parse::<f64>()
                            .map_err(|e| anyhow!("pickle: bad FLOAT '{line}': {e}"))?,
                    ));
                },

                // --- strings and bytes ---
                b'X' => {
                    let len = self.read_u32()? as usize;
                    let value = self.read_utf8(len)?;
                    self.stack.push(PickleValue::String(value));
                },
                b'\x8c' => {
                    let len = self.read_u8()? as usize;
                    let value = self.read_utf8(len)?;
                    self.stack.push(PickleValue::String(value));
                },
                b'\x8d' => {
                    let len = self.read_u64()? as usize;
                    let value = self.read_utf8(len)?;
                    self.stack.push(PickleValue::String(value));
                },
                b'V' => {
                    let line = self.read_line()?;
                    self.stack.push(PickleValue::String(decode_raw_unicode_escape(&line)));
                },
                b'S' | b'T' | b'U' => {
                    let bytes = match opcode {
                        b'S' => {
                            let line = self.read_line()?;
                            strip_python_quotes(&line)?.into_bytes()
                        },
                        b'T' => {
                            let len = self.read_u32()? as usize;
                            self.read_bytes(len)?.to_vec()
                        },
                        _ => {
                            let len = self.read_u8()? as usize;
                            self.read_bytes(len)?.to_vec()
                        },
                    };
                    // Protocol 0-2 `str` is latin-1 on the wire.
                    let text: String = bytes.iter().map(|&b| b as char).collect();
                    self.stack.push(PickleValue::String(text));
                },
                b'B' => {
                    let len = self.read_u32()? as usize;
                    let bytes = self.read_bytes(len)?.to_vec();
                    self.stack.push(PickleValue::Bytes(bytes));
                },
                b'C' => {
                    let len = self.read_u8()? as usize;
                    let bytes = self.read_bytes(len)?.to_vec();
                    self.stack.push(PickleValue::Bytes(bytes));
                },
                b'\x8e' | b'\x96' => {
                    let len = self.read_u64()? as usize;
                    let bytes = self.read_bytes(len)?.to_vec();
                    self.stack.push(PickleValue::Bytes(bytes));
                },

                // --- containers ---
                b']' => self.stack.push(PickleValue::List(Vec::new())),
                b')' => self.stack.push(PickleValue::Tuple(Vec::new())),
                b'}' => self.stack.push(PickleValue::Dict(Vec::new())),
                b'\x8f' => self.stack.push(PickleValue::Set(Vec::new())),
                b'l' => {
                    let items = self.pop_mark()?;
                    self.stack.push(PickleValue::List(items));
                },
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(PickleValue::Tuple(items));
                },
                b'\x85' => {
                    let item = self.pop()?;
                    self.stack.push(PickleValue::Tuple(vec![item]));
                },
                b'\x86' => {
                    let second = self.pop()?;
                    let first = self.pop()?;
                    self.stack.push(PickleValue::Tuple(vec![first, second]));
                },
                b'\x87' => {
                    let third = self.pop()?;
                    let second = self.pop()?;
                    let first = self.pop()?;
                    self.stack.push(PickleValue::Tuple(vec![first, second, third]));
                },
                b'd' => {
                    let items = self.pop_mark()?;
                    if items.len() % 2 != 0 {
                        return Err(anyhow!("pickle: DICT with an odd number of items"));
                    }
                    let mut pairs = Vec::with_capacity(items.len() / 2);
                    let mut iterator = items.into_iter();
                    while let (Some(key), Some(value)) = (iterator.next(), iterator.next()) {
                        pairs.push((key, value));
                    }
                    self.stack.push(PickleValue::Dict(pairs));
                },
                b'\x91' => {
                    let items = self.pop_mark()?;
                    self.stack.push(PickleValue::Set(items));
                },

                // --- container mutation ---
                b'a' => {
                    let item = self.pop()?;
                    self.append_to_top(vec![item])?;
                },
                b'e' => {
                    let items = self.pop_mark()?;
                    self.append_to_top(items)?;
                },
                b'\x90' => {
                    let items = self.pop_mark()?;
                    self.append_to_top(items)?;
                },
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    self.set_items(vec![(key, value)])?;
                },
                b'u' => {
                    let items = self.pop_mark()?;
                    if items.len() % 2 != 0 {
                        return Err(anyhow!("pickle: SETITEMS with an odd number of items"));
                    }
                    let mut pairs = Vec::with_capacity(items.len() / 2);
                    let mut iterator = items.into_iter();
                    while let (Some(key), Some(value)) = (iterator.next(), iterator.next()) {
                        pairs.push((key, value));
                    }
                    self.set_items(pairs)?;
                },

                // --- memo ---
                b'q' => {
                    let index = self.read_u8()? as u64;
                    self.memoize(index)?;
                },
                b'r' => {
                    let index = self.read_u32()? as u64;
                    self.memoize(index)?;
                },
                b'p' => {
                    let line = self.read_line()?;
                    let index = line
                        .parse::<u64>()
                        .map_err(|e| anyhow!("pickle: bad PUT index '{line}': {e}"))?;
                    self.memoize(index)?;
                },
                b'\x94' => {
                    let index = self.memo.len() as u64;
                    self.memoize(index)?;
                },
                b'h' => {
                    let index = self.read_u8()? as u64;
                    self.recall(index)?;
                },
                b'j' => {
                    let index = self.read_u32()? as u64;
                    self.recall(index)?;
                },
                b'g' => {
                    let line = self.read_line()?;
                    let index = line
                        .parse::<u64>()
                        .map_err(|e| anyhow!("pickle: bad GET index '{line}': {e}"))?;
                    self.recall(index)?;
                },

                // --- globals, reduction, persistent ids ---
                b'c' => {
                    let module = self.read_line()?;
                    let name = self.read_line()?;
                    self.stack.push(PickleValue::Global { module, name });
                },
                b'\x93' => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    let module = module
                        .as_str()
                        .ok_or_else(|| anyhow!("pickle: STACK_GLOBAL module is not a string"))?
                        .to_string();
                    let name = name
                        .as_str()
                        .ok_or_else(|| anyhow!("pickle: STACK_GLOBAL name is not a string"))?
                        .to_string();
                    self.stack.push(PickleValue::Global { module, name });
                },
                b'R' => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    self.stack.push(reduce(callable, args));
                },
                b'\x81' => {
                    let args = self.pop()?;
                    let class = self.pop()?;
                    self.stack.push(reduce(class, args));
                },
                b'\x92' => {
                    let kwargs = self.pop()?;
                    let args = self.pop()?;
                    let class = self.pop()?;
                    let _ = kwargs;
                    self.stack.push(reduce(class, args));
                },
                b'b' => {
                    let state = self.pop()?;
                    let target = self.pop()?;
                    self.stack.push(build(target, state));
                },
                b'Q' => {
                    let id = self.pop()?;
                    self.stack.push(PickleValue::PersistentId(Box::new(id)));
                },
                b'P' => {
                    let line = self.read_line()?;
                    self.stack.push(PickleValue::PersistentId(Box::new(PickleValue::String(
                        line,
                    ))));
                },

                other => {
                    return Err(anyhow!(
                        "pickle: opcode 0x{other:02X} ({:?}) is not implemented by this reader",
                        other as char
                    ))
                },
            }
        }
    }

    fn pop(&mut self) -> Result<PickleValue> {
        self.stack.pop().ok_or_else(|| anyhow!("pickle: pop from an empty stack"))
    }

    fn pop_mark(&mut self) -> Result<Vec<PickleValue>> {
        let saved = self
            .marks
            .pop()
            .ok_or_else(|| anyhow!("pickle: opcode consumed a MARK that was never pushed"))?;
        Ok(std::mem::replace(&mut self.stack, saved))
    }

    fn memoize(&mut self, index: u64) -> Result<()> {
        let value = self
            .stack
            .last()
            .cloned()
            .ok_or_else(|| anyhow!("pickle: memo of an empty stack"))?;
        self.memo.insert(index, value);
        Ok(())
    }

    fn recall(&mut self, index: u64) -> Result<()> {
        let value = self
            .memo
            .get(&index)
            .cloned()
            .ok_or_else(|| anyhow!("pickle: memo slot {index} was never written"))?;
        self.stack.push(value);
        Ok(())
    }

    fn append_to_top(&mut self, items: Vec<PickleValue>) -> Result<()> {
        match self.stack.last_mut() {
            Some(PickleValue::List(existing)) | Some(PickleValue::Set(existing)) => {
                existing.extend(items);
                Ok(())
            },
            Some(other) => Err(anyhow!("pickle: cannot append to {other:?}")),
            None => Err(anyhow!("pickle: append with an empty stack")),
        }
    }

    fn set_items(&mut self, pairs: Vec<(PickleValue, PickleValue)>) -> Result<()> {
        match self.stack.last_mut() {
            Some(PickleValue::Dict(existing)) => {
                for (key, value) in pairs {
                    match existing.iter_mut().find(|(k, _)| *k == key) {
                        Some(slot) => slot.1 = value,
                        None => existing.push((key, value)),
                    }
                }
                Ok(())
            },
            Some(other) => Err(anyhow!("pickle: cannot set items on {other:?}")),
            None => Err(anyhow!("pickle: setitem with an empty stack")),
        }
    }

    fn read_u8(&mut self) -> Result<u8> {
        let byte = *self
            .bytes
            .get(self.position)
            .ok_or_else(|| anyhow!("pickle: stream ended mid-opcode"))?;
        self.position += 1;
        Ok(byte)
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| anyhow!("pickle: length overflow"))?;
        let slice = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| anyhow!("pickle: read of {len} bytes runs past the end"))?;
        self.position = end;
        Ok(slice)
    }

    fn read_utf8(&mut self, len: usize) -> Result<String> {
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| anyhow!("pickle: string is not valid UTF-8: {e}"))
    }

    fn read_u16(&mut self) -> Result<u16> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_be_u64(&mut self) -> Result<u64> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Little-endian two's complement integer of `len` bytes (`LONG1`/`LONG4`).
    fn read_signed_long(&mut self, len: usize) -> Result<i64> {
        if len == 0 {
            return Ok(0);
        }
        if len > 8 {
            return Err(anyhow!(
                "pickle: {len}-byte integer does not fit in i64; this reader handles integers up \
                 to 64 bits"
            ));
        }
        let bytes = self.read_bytes(len)?;
        let negative = bytes[len - 1] & 0x80 != 0;
        let mut buffer = if negative { [0xFFu8; 8] } else { [0u8; 8] };
        buffer[..len].copy_from_slice(bytes);
        Ok(i64::from_le_bytes(buffer))
    }

    fn read_line(&mut self) -> Result<String> {
        let start = self.position;
        while self.position < self.bytes.len() && self.bytes[self.position] != b'\n' {
            self.position += 1;
        }
        if self.position >= self.bytes.len() {
            return Err(anyhow!("pickle: unterminated text argument"));
        }
        let line = &self.bytes[start..self.position];
        self.position += 1;
        Ok(line.iter().map(|&b| b as char).collect())
    }
}

/// Record a `callable(*args)` without performing it.
///
/// The two dictionary constructors are materialised, because `SETITEMS` is applied
/// to their result and a state dict is exactly that.
fn reduce(callable: PickleValue, args: PickleValue) -> PickleValue {
    if let PickleValue::Global { module, name } = &callable {
        let empty_args = matches!(&args, PickleValue::Tuple(items) if items.is_empty());
        let is_dict = (module == "collections" && (name == "OrderedDict" || name == "defaultdict"))
            || (module == "builtins" && name == "dict")
            || (module == "__builtin__" && name == "dict");
        if is_dict && empty_args {
            return PickleValue::Dict(Vec::new());
        }
    }
    PickleValue::Reduce {
        callable: Box::new(callable),
        args: Box::new(args),
    }
}

/// Apply `BUILD` state to a value.
fn build(target: PickleValue, state: PickleValue) -> PickleValue {
    match target {
        PickleValue::Reduce { callable, args } => PickleValue::Object {
            callable,
            args,
            state: Box::new(state),
        },
        // A dict already carries its contents; `BUILD` on it adds attributes we
        // have no use for.
        other => other,
    }
}

fn strip_python_quotes(line: &str) -> Result<String> {
    let trimmed = line.trim();
    let inner = trimmed
        .strip_prefix('\'')
        .and_then(|rest| rest.strip_suffix('\''))
        .or_else(|| trimmed.strip_prefix('"').and_then(|rest| rest.strip_suffix('"')))
        .ok_or_else(|| anyhow!("pickle: STRING argument is not quoted: {line}"))?;
    Ok(inner.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\"))
}

/// Decode the `raw-unicode-escape` form used by the protocol-0 `UNICODE` opcode.
fn decode_raw_unicode_escape(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(character) = chars.next() {
        if character != '\\' {
            out.push(character);
            continue;
        }
        match chars.peek() {
            Some('u') => {
                chars.next();
                let hex: String = chars.by_ref().take(4).collect();
                match u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                    Some(decoded) => out.push(decoded),
                    None => {
                        out.push('\\');
                        out.push('u');
                        out.push_str(&hex);
                    },
                }
            },
            _ => out.push('\\'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the pickle stream `{"a": 1, "b": [2, 3]}` byte by byte.
    fn simple_dict_pickle() -> Vec<u8> {
        let mut out = vec![0x80, 2]; // PROTO 2
        out.push(b'}'); // EMPTY_DICT
        out.push(b'q');
        out.push(0); // BINPUT 0
        out.push(b'('); // MARK
        out.extend_from_slice(&[b'X', 1, 0, 0, 0, b'a']); // BINUNICODE "a"
        out.extend_from_slice(&[b'K', 1]); // BININT1 1
        out.extend_from_slice(&[b'X', 1, 0, 0, 0, b'b']); // BINUNICODE "b"
        out.push(b']'); // EMPTY_LIST
        out.push(b'('); // MARK
        out.extend_from_slice(&[b'K', 2, b'K', 3]);
        out.push(b'e'); // APPENDS
        out.push(b'u'); // SETITEMS
        out.push(b'.'); // STOP
        out
    }

    #[test]
    fn loads_a_hand_built_dict() {
        let value = load(&simple_dict_pickle()).expect("load");
        assert_eq!(value.get("a").and_then(PickleValue::as_int), Some(1));
        assert_eq!(
            value.get("b").and_then(PickleValue::as_sequence),
            Some(&[PickleValue::Int(2), PickleValue::Int(3)][..])
        );
    }

    #[test]
    fn memo_round_trips() {
        // {"a": 1} with the key stored in the memo and read back as the value.
        let mut out = vec![0x80, 2, b'}'];
        out.extend_from_slice(&[b'X', 1, 0, 0, 0, b'a']);
        out.extend_from_slice(&[b'q', 7]); // BINPUT 7
        out.extend_from_slice(&[b'h', 7]); // BINGET 7
        out.push(b's'); // SETITEM
        out.push(b'.');

        let value = load(&out).expect("load");
        assert_eq!(value.get("a").and_then(PickleValue::as_str), Some("a"));
    }

    #[test]
    fn stack_global_and_reduce_are_recorded_not_executed() {
        let mut out = vec![0x80, 4];
        out.extend_from_slice(&[b'\x8c', 11]);
        out.extend_from_slice(b"torch._utils");
        // Correct the length prefix for the module name.
        let module = b"torch._utils";
        out.truncate(2);
        out.push(b'\x8c');
        out.push(module.len() as u8);
        out.extend_from_slice(module);
        let name = b"_rebuild_tensor_v2";
        out.push(b'\x8c');
        out.push(name.len() as u8);
        out.extend_from_slice(name);
        out.push(b'\x93'); // STACK_GLOBAL
        out.push(b')'); // EMPTY_TUPLE
        out.push(b'R'); // REDUCE
        out.push(b'.');

        let value = load(&out).expect("load");
        match value {
            PickleValue::Reduce { callable, args } => {
                assert_eq!(
                    *callable,
                    PickleValue::Global {
                        module: "torch._utils".to_string(),
                        name: "_rebuild_tensor_v2".to_string()
                    }
                );
                assert_eq!(*args, PickleValue::Tuple(Vec::new()));
            },
            other => panic!("expected a Reduce, got {other:?}"),
        }
    }

    #[test]
    fn ordered_dict_reduce_becomes_a_dict_so_setitems_works() {
        let mut out = vec![0x80, 4];
        let module = b"collections";
        out.push(b'\x8c');
        out.push(module.len() as u8);
        out.extend_from_slice(module);
        let name = b"OrderedDict";
        out.push(b'\x8c');
        out.push(name.len() as u8);
        out.extend_from_slice(name);
        out.push(b'\x93');
        out.push(b')');
        out.push(b'R');
        out.push(b'(');
        out.extend_from_slice(&[b'\x8c', 1, b'k']);
        out.extend_from_slice(&[b'K', 5]);
        out.push(b'u'); // SETITEMS
        out.push(b'.');

        let value = load(&out).expect("load");
        assert_eq!(value.get("k").and_then(PickleValue::as_int), Some(5));
    }

    #[test]
    fn persistent_ids_are_captured() {
        let mut out = vec![0x80, 2, b'('];
        out.extend_from_slice(&[b'X', 7, 0, 0, 0]);
        out.extend_from_slice(b"storage");
        out.extend_from_slice(&[b'K', 3]);
        out.push(b't'); // TUPLE
        out.push(b'Q'); // BINPERSID
        out.push(b'.');

        let value = load(&out).expect("load");
        match value {
            PickleValue::PersistentId(inner) => {
                assert_eq!(
                    inner.as_sequence(),
                    Some(&[PickleValue::String("storage".into()), PickleValue::Int(3)][..])
                );
            },
            other => panic!("expected a PersistentId, got {other:?}"),
        }
    }

    #[test]
    fn long1_decodes_negative_values() {
        // LONG1 with two bytes, 0xFF 0xFF = -1
        let bytes = vec![0x80, 2, b'\x8a', 2, 0xFF, 0xFF, b'.'];
        assert_eq!(load(&bytes).expect("load"), PickleValue::Int(-1));

        // LONG1 with one byte, 0x7F = 127
        let bytes = vec![0x80, 2, b'\x8a', 1, 0x7F, b'.'];
        assert_eq!(load(&bytes).expect("load"), PickleValue::Int(127));
    }

    #[test]
    fn binfloat_is_big_endian() {
        let mut bytes = vec![0x80, 2, b'G'];
        bytes.extend_from_slice(&1.5f64.to_be_bytes());
        bytes.push(b'.');
        assert_eq!(load(&bytes).expect("load"), PickleValue::Float(1.5));
    }

    #[test]
    fn frames_are_skipped() {
        let mut bytes = vec![0x80, 4, b'\x95'];
        bytes.extend_from_slice(&7u64.to_le_bytes());
        bytes.extend_from_slice(&[b'K', 42, b'.']);
        assert_eq!(load(&bytes).expect("load"), PickleValue::Int(42));
    }

    #[test]
    fn unsupported_opcodes_are_named() {
        // 0x99 is not a pickle opcode this reader implements.
        let bytes = vec![0x80, 5, 0x99, b'.'];
        let err = load(&bytes).expect_err("must fail");
        assert!(err.to_string().contains("0x99"), "{err}");
    }

    #[test]
    fn truncated_streams_are_rejected() {
        let mut bytes = simple_dict_pickle();
        bytes.truncate(bytes.len() - 3);
        assert!(load(&bytes).is_err());
    }

    #[test]
    fn newer_protocols_are_rejected_clearly() {
        let bytes = vec![0x80, 9, b'.'];
        let err = load(&bytes).expect_err("must fail");
        assert!(err.to_string().contains("protocol 9"), "{err}");
    }
}
