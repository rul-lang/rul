use std::io::{self, Read, Write};

fn fib(n: i64) -> i64 {
    if n < 2 {
        return n;
    }
    fib(n - 1) + fib(n - 2)
}

fn main() {
    let mut buf = [0u8; 8];
    io::stdin().read_exact(&mut buf).expect("failed to read n");
    let n = i64::from_le_bytes(buf);
    let res = fib(n);
    io::stdout()
        .write_all(&res.to_le_bytes())
        .expect("failed to write result");
}
