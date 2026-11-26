use std::collections::HashMap;
use std::fmt::Write as _;
use std::hash::{BuildHasher, Hasher};
use std::io::{self, Read, Write};

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

struct Fnv1aHasher(u64);

impl Hasher for Fnv1aHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(FNV_PRIME);
        }
        self.0 = h;
    }
}

#[derive(Clone, Default)]
struct Fnv1aBuildHasher;

impl BuildHasher for Fnv1aBuildHasher {
    type Hasher = Fnv1aHasher;

    fn build_hasher(&self) -> Self::Hasher {
        Fnv1aHasher(FNV_OFFSET_BASIS)
    }
}

fn parse_i64_bytes(bytes: &[u8]) -> i64 {
    if bytes.is_empty() {
        return 0;
    }
    let mut i = 0usize;
    let mut neg = false;
    if bytes[0] == b'-' {
        neg = true;
        i = 1;
    } else if bytes[0] == b'+' {
        i = 1;
    }
    let mut acc: i64 = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if !b.is_ascii_digit() {
            break;
        }
        acc = acc.wrapping_mul(10).wrapping_sub(i64::from(b - b'0'));
        i += 1;
    }
    if neg {
        acc
    } else {
        acc.wrapping_neg()
    }
}

fn main() {
    let mut input = Vec::new();
    io::stdin()
        .read_to_end(&mut input)
        .expect("failed to read stdin");

    let mut map: HashMap<&[u8], i64, Fnv1aBuildHasher> =
        HashMap::with_hasher(Fnv1aBuildHasher);
    let mut out = String::with_capacity(input.len() / 2 + 64);

    let mut pos = 0usize;
    let len = input.len();
    while pos < len {
        let line_start = pos;
        while pos < len && input[pos] != b'\n' {
            pos += 1;
        }
        let mut line_end = pos;
        if pos < len {
            pos += 1;
        }
        if line_end > line_start && input[line_end - 1] == b'\r' {
            line_end -= 1;
        }
        if line_end <= line_start {
            continue;
        }
        let line = &input[line_start..line_end];
        match line[0] {
            b'S' if line.len() > 4 && &line[0..4] == b"SET " => {
                let rest = &line[4..];
                if let Some(sp) = rest.iter().position(|&b| b == b' ') {
                    let key = &rest[..sp];
                    let val = parse_i64_bytes(&rest[sp + 1..]);
                    map.insert(key, val);
                    out.push_str("OK\n");
                }
            }
            b'G' if line.len() > 4 && &line[0..4] == b"GET " => {
                let key = &line[4..];
                if let Some(&v) = map.get(key) {
                    let _ = writeln!(out, "{v}");
                } else {
                    out.push_str("MISS\n");
                }
            }
            b'D' if line.len() > 4 && &line[0..4] == b"DEL " => {
                let key = &line[4..];
                if map.remove(key).is_some() {
                    out.push_str("OK\n");
                } else {
                    out.push_str("MISS\n");
                }
            }
            b'C' if line == b"COUNT" => {
                let _ = writeln!(out, "{}", map.len());
            }
            _ => {}
        }
    }

    io::stdout()
        .write_all(out.as_bytes())
        .expect("failed to write stdout");
}
