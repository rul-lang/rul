use std::io::{self, Read, Write};

fn mark_multiples(flags: &mut [u8], p: usize, limit: usize) {
    let mut m = p * p;
    while m <= limit {
        flags[m] = 1;
        m += p;
    }
}

fn count_primes(flags: &[u8], limit: usize) -> i64 {
    let mut cnt: i64 = 0;
    let mut i = 2usize;
    while i <= limit {
        if flags[i] == 0 {
            cnt += 1;
        }
        i += 1;
    }
    cnt
}

fn main() {
    let mut buf = [0u8; 8];
    io::stdin().read_exact(&mut buf).expect("failed to read N");
    let limit = i64::from_le_bytes(buf) as usize;

    let mut flags = vec![0u8; limit + 1];
    let mut p = 2usize;
    while p * p <= limit {
        if flags[p] == 0 {
            mark_multiples(&mut flags, p, limit);
        }
        p += 1;
    }
    let count = count_primes(&flags, limit);
    io::stdout()
        .write_all(&count.to_le_bytes())
        .expect("failed to write prime count");
}
