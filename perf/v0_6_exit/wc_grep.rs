use std::io::{self, Read, Write};

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

fn is_wc_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C)
}

fn find_substr(hay: &[u8], pat: &[u8]) -> Option<usize> {
    if pat.is_empty() {
        return Some(0);
    }
    if pat.len() > hay.len() {
        return None;
    }
    let first = pat[0];
    let last_idx = pat.len() - 1;
    let last = pat[last_idx];
    let limit = hay.len() - pat.len() + 1;
    for i in 0..limit {
        if hay[i] == first && hay[i + last_idx] == last {
            let mut ok = true;
            for j in 1..last_idx {
                if hay[i + j] != pat[j] {
                    ok = false;
                    break;
                }
            }
            if ok {
                return Some(i);
            }
        }
    }
    None
}

fn main() {
    let mut input = Vec::new();
    io::stdin()
        .read_to_end(&mut input)
        .expect("failed to read stdin");

    let (pat, text) = match input.iter().position(|&b| b == b'\n') {
        Some(nl) => (&input[..nl], &input[nl + 1..]),
        None => (&input[..], &[][..]),
    };

    let bytes = text.len() as i64;
    let mut lines: i64 = 0;
    let mut words: i64 = 0;
    let mut in_word = false;
    let mut matches: i64 = 0;
    let mut digest: u64 = FNV_OFFSET_BASIS;

    let mut pos = 0usize;
    let tlen = text.len();
    while pos < tlen {
        let line_start = pos;
        while pos < tlen {
            let b = text[pos];
            if is_wc_space(b) {
                in_word = false;
            } else if !in_word {
                in_word = true;
                words += 1;
            }
            if b == b'\n' {
                lines += 1;
                break;
            }
            pos += 1;
        }
        let line_end = pos;
        if pos < tlen && text[pos] == b'\n' {
            pos += 1;
        }
        let line = &text[line_start..line_end];
        if find_substr(line, pat).is_some() {
            matches += 1;
            for &b in line {
                digest ^= u64::from(b);
                digest = digest.wrapping_mul(FNV_PRIME);
            }
            digest ^= u64::from(b'\n');
            digest = digest.wrapping_mul(FNV_PRIME);
        }
    }

    let digest_out = (digest as i64) & 0x7FFF_FFFF_FFFF_FFFF;
    let out = format!(
        "lines={lines} words={words} bytes={bytes} matches={matches} digest={digest_out}\n"
    );
    io::stdout()
        .write_all(out.as_bytes())
        .expect("failed to write stdout");
}
