use std::io::{self, Read, Write};

const K: [u32; 64] = [
    0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5, 0x3956_c25b, 0x59f1_11f1, 0x923f_82a4,
    0xab1c_5ed5, 0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3, 0x72be_5d74, 0x80de_b1fe,
    0x9bdc_06a7, 0xc19b_f174, 0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc, 0x2de9_2c6f,
    0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da, 0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7,
    0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967, 0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc,
    0x5338_0d13, 0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85, 0xa2bf_e8a1, 0xa81a_664b,
    0xc24b_8b70, 0xc76c_51a3, 0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070, 0x19a4_c116,
    0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5, 0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
    0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208, 0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7,
    0xc671_78f2,
];

const H0: [u32; 8] = [
    0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a, 0x510e_527f, 0x9b05_688c, 0x1f83_d9ab,
    0x5be0_cd19,
];

fn compress_block(h: &mut [u32; 8], w: &mut [u32; 64], block: &[u8]) {
    for i in 0..16 {
        let off = i * 4;
        w[i] = ((block[off] as u32) << 24)
            | ((block[off + 1] as u32) << 16)
            | ((block[off + 2] as u32) << 8)
            | (block[off + 3] as u32);
    }
    for i in 16..64 {
        let w15 = w[i - 15];
        let s0 = w15.rotate_right(7) ^ w15.rotate_right(18) ^ (w15 >> 3);
        let w2 = w[i - 2];
        let s1 = w2.rotate_right(17) ^ w2.rotate_right(19) ^ (w2 >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let mut a = h[0];
    let mut b = h[1];
    let mut c = h[2];
    let mut d = h[3];
    let mut e = h[4];
    let mut f = h[5];
    let mut g = h[6];
    let mut hh = h[7];

    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = hh
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
    h[4] = h[4].wrapping_add(e);
    h[5] = h[5].wrapping_add(f);
    h[6] = h[6].wrapping_add(g);
    h[7] = h[7].wrapping_add(hh);
}

fn main() {
    let mut msg = Vec::new();
    io::stdin()
        .read_to_end(&mut msg)
        .expect("failed to read message");

    let mut h = H0;
    let mut w = [0u32; 64];

    let full_end = msg.len() & !63;
    let mut off = 0usize;
    while off < full_end {
        compress_block(&mut h, &mut w, &msg[off..off + 64]);
        off += 64;
    }

    let rem = msg.len() - full_end;
    let tail_blocks = if rem < 56 { 1 } else { 2 };
    let tail_len = tail_blocks * 64;
    let mut tail = [0u8; 128];
    tail[..rem].copy_from_slice(&msg[full_end..]);
    tail[rem] = 0x80;
    let bit_len = (msg.len() as u64) * 8;
    tail[tail_len - 8..tail_len].copy_from_slice(&bit_len.to_be_bytes());

    compress_block(&mut h, &mut w, &tail[0..64]);
    if tail_blocks == 2 {
        compress_block(&mut h, &mut w, &tail[64..128]);
    }

    let mut digest = [0u8; 32];
    for (k, &word) in h.iter().enumerate() {
        digest[k * 4..k * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    io::stdout()
        .write_all(&digest)
        .expect("failed to write digest");
}
