use std::convert::TryInto;
use std::io::{self, Read, Write};

fn partition(a: &mut [i64], lo: usize, hi: usize) -> usize {
    let pivot = a[hi];
    let mut i = lo;
    let mut j = lo;
    while j < hi {
        if a[j] <= pivot {
            a.swap(i, j);
            i += 1;
        }
        j += 1;
    }
    a.swap(i, hi);
    i
}

fn qsort(a: &mut [i64], lo: isize, hi: isize) {
    if lo >= hi {
        return;
    }
    let p = partition(a, lo as usize, hi as usize) as isize;
    qsort(a, lo, p - 1);
    qsort(a, p + 1, hi);
}

fn main() {
    let mut buf = [0u8; 16];
    io::stdin()
        .read_exact(&mut buf)
        .expect("failed to read count and seed");
    let count = i64::from_le_bytes(buf[0..8].try_into().expect("count slice")) as usize;
    let mut x = i64::from_le_bytes(buf[8..16].try_into().expect("seed slice"));

    let mut a = vec![0i64; count];
    for slot in &mut a {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005_i64)
            .wrapping_add(1_442_695_040_888_963_407_i64);
        *slot = x;
    }

    if count > 1 {
        qsort(&mut a, 0, (count as isize) - 1);
    }

    let mut acc: i64 = 0;
    for (idx, &val) in a.iter().enumerate() {
        acc = acc.wrapping_add(val.wrapping_mul((idx as i64) + 1));
    }

    io::stdout()
        .write_all(&acc.to_le_bytes())
        .expect("failed to write checksum");
}
