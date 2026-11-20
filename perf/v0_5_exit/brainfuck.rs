use std::io::{self, Read, Write};

fn find_code_end(input: &[u8]) -> usize {
    for (i, &b) in input.iter().enumerate() {
        if b == b'!' {
            return i;
        }
    }
    input.len()
}

fn build_jumps(code: &[u8]) -> Vec<usize> {
    let mut jumps = vec![0usize; code.len()];
    let mut stack = Vec::with_capacity(code.len());
    for (i, &ch) in code.iter().enumerate() {
        if ch == b'[' {
            stack.push(i);
        } else if ch == b']' {
            let open = stack.pop().expect("unmatched ']'");
            jumps[open] = i;
            jumps[i] = open;
        }
    }
    jumps
}

fn main() {
    let mut input = Vec::new();
    io::stdin()
        .read_to_end(&mut input)
        .expect("failed to read program");

    let total_len = input.len();
    let code_len = find_code_end(&input);
    let jumps = build_jumps(&input[..code_len]);
    let mut tape = vec![0u8; 30_000];
    let mut output = Vec::new();

    let mut pc = 0usize;
    let mut dp = 0usize;
    let mut in_pos = code_len + 1;

    while pc < code_len {
        let cmd = input[pc];
        if cmd == b'>' {
            dp += 1;
            if dp == 30_000 {
                dp = 0;
            }
        } else if cmd == b'<' {
            if dp == 0 {
                dp = 29_999;
            } else {
                dp -= 1;
            }
        } else if cmd == b'+' {
            tape[dp] = tape[dp].wrapping_add(1);
        } else if cmd == b'-' {
            tape[dp] = tape[dp].wrapping_sub(1);
        } else if cmd == b']' {
            if tape[dp] != 0 {
                pc = jumps[pc];
            }
        } else if cmd == b'[' {
            if tape[dp] == 0 {
                pc = jumps[pc];
            }
        } else if cmd == b'.' {
            output.push(tape[dp]);
        } else if cmd == b',' {
            if in_pos < total_len {
                tape[dp] = input[in_pos];
                in_pos += 1;
            } else {
                tape[dp] = 0;
            }
        }
        pc += 1;
    }

    io::stdout()
        .write_all(&output)
        .expect("failed to write output");
}
