use clap::{ArgAction, Parser, ValueEnum};
use rand::prelude::*;
use std::f32::consts::TAU;
use std::fs::File;
use std::io::{BufWriter, Write};

/// Default luminance quantization table (quality 50), natural order.
const STD_LUMA_QTABLE_Q50_NATURAL: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
    56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104,
    113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Default chrominance quantization table (quality 50), natural order (fixed to 64).
const STD_CHROMA_QTABLE_Q50_NATURAL: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, //
    18, 21, 26, 66, 99, 99, 99, 99, //
    24, 26, 56, 99, 99, 99, 99, 99, //
    47, 66, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99,
];

/// Zig-zag order mapping: index in zig-zag -> index in natural (row-major).
const ZZ_TO_NAT: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
    20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58,
    59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Standard Huffman tables (baseline).
/// DC luminance: bits must sum to 12.
const STD_BITS_DC_LUMA: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const STD_VALS_DC_LUMA: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// AC luminance: standard (sums to 162).
const STD_BITS_AC_LUMA: [u8; 16] = [
    0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
];
const STD_VALS_AC_LUMA: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum FreqMode {
    Low,
    All,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Pattern {
    Random,
    Waves,
}

/// CLI
#[derive(Parser, Debug)]
#[command(name = "randjpeg", about = "Generate random-but-structured JPEG")]
struct Args {
    /// Output JPEG path
    #[arg(short, long)]
    output: String,

    /// Width in pixels (>= 8)
    #[arg(short = 'w', long, default_value_t = 512)]
    width: u16,

    /// Height in pixels (>= 8). -h is help, so use -H for height.
    #[arg(short = 'H', long, default_value_t = 512)]
    height: u16,

    /// Random seed (optional)
    #[arg(long)]
    seed: Option<u64>,

    /// Number of active AC coefficients:
    /// - random: approximate non-zeros per block
    /// - waves: number of global AC frequencies to activate
    #[arg(long, default_value_t = 4)]
    nonzero: usize,

    /// Max AC bit-size (1..10). Larger => stronger amplitudes.
    #[arg(long, default_value_t = 6)]
    max_ac_bits: u8,

    /// Frequency selection mode (which AC positions to use)
    #[arg(long, value_enum, default_value = "low")]
    freq: FreqMode,

    /// If freq=low, up to which zig-zag index (1..63) to draw from
    #[arg(long, default_value_t = 12)]
    low_span: usize,

    /// Quality (1..100). Scales the quantization tables.
    #[arg(long, default_value_t = 50)]
    quality: u8,

    /// If set, DC uses a tiny random walk (adds slow gradients)
    #[arg(long, action = ArgAction::SetTrue)]
    dc_random_walk: bool,

    /// Force grayscale (1 component). Default is color (YCbCr 4:4:4).
    #[arg(long, action = ArgAction::SetTrue)]
    grayscale: bool,

    /// Pattern of coefficients
    #[arg(long, value_enum, default_value = "waves")]
    pattern: Pattern,

    /// Number of sine waves mixed per active coefficient (pattern=waves)
    #[arg(long, default_value_t = 3)]
    num_waves: usize,
}

#[derive(Clone, Copy)]
struct HuffCode {
    code: u16,
    len: u8,
}

struct HuffTable {
    codes: [Option<HuffCode>; 256],
}

impl HuffTable {
    fn from_spec(bits: &[u8; 16], vals: &[u8]) -> Self {
        let mut codes = [None; 256];

        let expected: usize = bits.iter().map(|&b| b as usize).sum();
        assert!(
            expected == vals.len(),
            "Huffman bits/vals mismatch: bits sum={}, vals={}",
            expected,
            vals.len()
        );

        let mut code: u16 = 0;
        let mut k: usize = 0;
        for i in 0..16 {
            let n = bits[i] as usize;
            for _ in 0..n {
                let sym = vals[k] as usize;
                codes[sym] = Some(HuffCode {
                    code,
                    len: (i as u8) + 1,
                });
                code += 1;
                k += 1;
            }
            code <<= 1;
        }
        Self { codes }
    }

    fn get(&self, sym: u8) -> HuffCode {
        self.codes[sym as usize].expect("Missing Huffman symbol")
    }
}

struct BitWriter<W: Write> {
    w: W,
    acc: u32,
    nbits: u8,
}

impl<W: Write> BitWriter<W> {
    fn new(w: W) -> Self {
        Self {
            w,
            acc: 0,
            nbits: 0,
        }
    }

    fn write_u8(&mut self, b: u8) -> std::io::Result<()> {
        self.w.write_all(&[b])
    }

    fn write_u16_be(&mut self, v: u16) -> std::io::Result<()> {
        self.w.write_all(&[(v >> 8) as u8, (v & 0xFF) as u8])
    }

    fn write_marker(&mut self, m: u8) -> std::io::Result<()> {
        self.write_u8(0xFF)?;
        self.write_u8(m)
    }

    fn write_stuffed_byte(&mut self, b: u8) -> std::io::Result<()> {
        self.w.write_all(&[b])?;
        if b == 0xFF {
            self.w.write_all(&[0x00])?;
        }
        Ok(())
    }

    fn put_bits(&mut self, code: u16, len: u8) -> std::io::Result<()> {
        if len == 0 {
            return Ok(());
        }
        self.acc = (self.acc << len) | (code as u32 & ((1u32 << len) - 1));
        self.nbits += len;
        while self.nbits >= 8 {
            let out = (self.acc >> (self.nbits - 8)) as u8;
            self.write_stuffed_byte(out)?;
            self.nbits -= 8;
        }
        Ok(())
    }

    fn put_huff(&mut self, ht: &HuffTable, sym: u8) -> std::io::Result<()> {
        let hc = ht.get(sym);
        self.put_bits(hc.code, hc.len)
    }

    fn flush_to_byte(&mut self) -> std::io::Result<()> {
        if self.nbits > 0 {
            let out =
                (self.acc << (8 - self.nbits)) as u8 | ((1u8 << (8 - self.nbits)) - 1);
            self.write_stuffed_byte(out)?;
            self.nbits = 0;
        }
        Ok(())
    }
}

fn scaled_qtable_natural(base: &[u8; 64], quality: u8) -> [u8; 64] {
    let q = if quality == 0 { 1 } else { quality.min(100) };
    let scale: i32 = if q < 50 {
        5000 / (q as i32)
    } else {
        200 - 2 * q as i32
    };
    let mut out = [0u8; 64];
    for (i, b) in base.iter().enumerate() {
        let mut v = ((*b as i32) * scale + 50) / 100;
        v = v.clamp(1, 255);
        out[i] = v as u8;
    }
    out
}

fn natural_to_zz(natural: &[u8; 64]) -> [u8; 64] {
    let mut zz = [0u8; 64];
    for (zz_idx, nat_idx) in ZZ_TO_NAT.iter().enumerate() {
        zz[zz_idx] = natural[*nat_idx];
    }
    zz
}

fn category(v: i32) -> u8 {
    if v == 0 {
        0
    } else {
        let m = v.abs();
        let msb = 31 - m.leading_zeros();
        (msb as u8) + 1
    }
}

fn amplitude_bits(v: i32, s: u8) -> u16 {
    if s == 0 {
        return 0;
    }
    if v >= 0 {
        v as u16
    } else {
        let mask = (1u16 << s) - 1;
        ((v + (mask as i32)) as u16) & mask
    }
}

fn write_soi<W: Write>(bw: &mut BitWriter<W>) -> std::io::Result<()> {
    bw.write_marker(0xD8)
}

fn write_app0_jfif<W: Write>(bw: &mut BitWriter<W>) -> std::io::Result<()> {
    bw.write_marker(0xE0)?;
    bw.write_u16_be(16)?;
    bw.w.write_all(b"JFIF\0")?;
    bw.write_u8(1)?;
    bw.write_u8(2)?;
    bw.write_u8(1)?;
    bw.write_u16_be(72)?;
    bw.write_u16_be(72)?;
    bw.write_u8(0)?;
    bw.write_u8(0)?;
    Ok(())
}

fn write_dqt<W: Write>(
    bw: &mut BitWriter<W>,
    qzz: &[u8; 64],
    tq: u8,
) -> std::io::Result<()> {
    bw.write_marker(0xDB)?;
    bw.write_u16_be(67)?;
    bw.write_u8((0 << 4) | (tq & 0x0F))?;
    for &v in qzz {
        bw.write_u8(v)?;
    }
    Ok(())
}

fn write_sof0_gray<W: Write>(
    bw: &mut BitWriter<W>,
    width: u16,
    height: u16,
    tq: u8,
) -> std::io::Result<()> {
    bw.write_marker(0xC0)?;
    bw.write_u16_be(11)?;
    bw.write_u8(8)?;
    bw.write_u16_be(height)?;
    bw.write_u16_be(width)?;
    bw.write_u8(1)?;
    bw.write_u8(1)?; // Y
    bw.write_u8((1 << 4) | 1)?; // H=1,V=1
    bw.write_u8(tq & 0x0F)?;
    Ok(())
}

fn write_sof0_ycc444<W: Write>(
    bw: &mut BitWriter<W>,
    width: u16,
    height: u16,
    tq_y: u8,
    tq_c: u8,
) -> std::io::Result<()> {
    bw.write_marker(0xC0)?;
    let nf = 3u8;
    let lf = 8 + 3 * (nf as u16);
    bw.write_u16_be(lf)?;
    bw.write_u8(8)?;
    bw.write_u16_be(height)?;
    bw.write_u16_be(width)?;
    bw.write_u8(nf)?;
    // Y
    bw.write_u8(1)?; // C1=1
    bw.write_u8((1 << 4) | 1)?; // H=1, V=1
    bw.write_u8(tq_y & 0x0F)?;
    // Cb
    bw.write_u8(2)?; // C2=2
    bw.write_u8((1 << 4) | 1)?;
    bw.write_u8(tq_c & 0x0F)?;
    // Cr
    bw.write_u8(3)?; // C3=3
    bw.write_u8((1 << 4) | 1)?;
    bw.write_u8(tq_c & 0x0F)?;
    Ok(())
}

fn write_dht_single<W: Write>(
    bw: &mut BitWriter<W>,
    bits: &[u8; 16],
    vals: &[u8],
    class: u8,
    tid: u8,
) -> std::io::Result<()> {
    bw.write_marker(0xC4)?;
    let nvals = vals.len() as u16;
    let len = 2 + 1 + 16 + nvals;
    bw.write_u16_be(len)?;
    bw.write_u8(((class & 1) << 4) | (tid & 0x0F))?;
    for &b in bits {
        bw.write_u8(b)?;
    }
    bw.w.write_all(vals)?;
    Ok(())
}

// SOS for 3 components (Y, Cb, Cr): Y uses tables 0/0, Cb+Cr use 1/1
fn write_sos_ycc444<W: Write>(bw: &mut BitWriter<W>) -> std::io::Result<()> {
    bw.write_marker(0xDA)?;
    bw.write_u16_be(12)?;
    bw.write_u8(3)?; // Nc
    bw.write_u8(1)?; // Y
    bw.write_u8((0 << 4) | 0)?;
    bw.write_u8(2)?; // Cb
    bw.write_u8((1 << 4) | 1)?;
    bw.write_u8(3)?; // Cr
    bw.write_u8((1 << 4) | 1)?;
    bw.write_u8(0)?; // Ss
    bw.write_u8(63)?; // Se
    bw.write_u8(0)?; // Ah/Al
    Ok(())
}

fn write_sos_gray<W: Write>(bw: &mut BitWriter<W>, td: u8, ta: u8) -> std::io::Result<()> {
    bw.write_marker(0xDA)?;
    bw.write_u16_be(8)?;
    bw.write_u8(1)?;
    bw.write_u8(1)?;
    bw.write_u8(((td & 0x0F) << 4) | (ta & 0x0F))?;
    bw.write_u8(0)?;
    bw.write_u8(63)?;
    bw.write_u8(0)?;
    Ok(())
}

fn write_eoi<W: Write>(bw: &mut BitWriter<W>) -> std::io::Result<()> {
    bw.write_marker(0xD9)
}

// -------- Pattern generation helpers (for "waves") --------

#[derive(Clone, Copy)]
struct Wave {
    kx: f32,    // cycles per block in x
    ky: f32,    // cycles per block in y
    phase: f32, // radians
    amp: f32,   // positive weight
}

fn make_waves<R: Rng>(
    rng: &mut R,
    num: usize,
    blocks_x: usize,
    blocks_y: usize,
) -> Vec<Wave> {
    let mut waves = Vec::with_capacity(num);
    let max_extent = blocks_x.max(blocks_y).max(8) as f32;
    for _ in 0..num {
        let theta = rng.gen_range(0.0..TAU);
        // Choose a period between 4 blocks and ~1/2 of the image
        let min_p = 4.0;
        let max_p = (max_extent * 0.5).max(min_p + 1.0);
        let period = rng.gen_range(min_p..max_p);
        let k = 1.0 / period; // cycles per block
        let kx = k * theta.cos();
        let ky = k * theta.sin();
        let phase = rng.gen_range(0.0..TAU);
        let amp = rng.gen_range(0.5..1.0);
        waves.push(Wave {
            kx,
            ky,
            phase,
            amp,
        });
    }
    waves
}

fn sample_waves(waves: &[Wave], bx: f32, by: f32) -> f32 {
    let mut s = 0.0f32;
    let mut wsum = 0.0f32;
    for w in waves {
        s += w.amp * ((w.kx * bx + w.ky * by) * TAU + w.phase).sin();
        wsum += w.amp.abs();
    }
    if wsum > 0.0 {
        (s / wsum).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn pick_active_indices<R: Rng>(
    rng: &mut R,
    nonzero: usize,
    freq: FreqMode,
    low_span: usize,
) -> Vec<usize> {
    let mut pool: Vec<usize> = match freq {
        FreqMode::Low => {
            let span = low_span.clamp(1, 63);
            (1..=span).collect()
        }
        FreqMode::All => (1..64).collect(),
    };
    pool.shuffle(rng);
    pool.truncate(nonzero.min(pool.len()));
    pool
}

fn generate_block_waves(
    ac_indices: &[usize],
    s_bits: &[u8],
    waves_per_index: &[Vec<Wave>],
    bx: usize,
    by: usize,
    dc_val: i32,
) -> [i16; 64] {
    debug_assert_eq!(ac_indices.len(), s_bits.len());
    debug_assert_eq!(ac_indices.len(), waves_per_index.len());

    let mut block_zz = [0i16; 64];
    block_zz[0] = dc_val as i16;

    let bx = bx as f32;
    let by = by as f32;

    for (j, &zz_idx) in ac_indices.iter().enumerate() {
        let s = s_bits[j].clamp(1, 10);
        let max_mag = ((1i32 << s) - 1) as f32;
        let val = sample_waves(&waves_per_index[j], bx, by);
        let v = (val * max_mag).round().clamp(-(max_mag), max_mag) as i16;
        block_zz[zz_idx] = v;
    }
    block_zz
}

// -------- Entropy block encoding --------

fn encode_block<W: Write>(
    bw: &mut BitWriter<W>,
    dc_ht: &HuffTable,
    ac_ht: &HuffTable,
    prev_dc: &mut i32,
    block_zz: &[i16; 64],
) -> std::io::Result<()> {
    // DC
    let dc = block_zz[0] as i32;
    let diff = dc - *prev_dc;
    *prev_dc = dc;
    let s = category(diff);
    bw.put_huff(dc_ht, s)?;
    if s > 0 {
        let bits = amplitude_bits(diff, s);
        bw.put_bits(bits, s)?;
    }

    // AC
    let mut run = 0u8;
    for i in 1..64 {
        let v = block_zz[i] as i32;
        if v == 0 {
            run += 1;
            if run == 16 {
                bw.put_huff(ac_ht, 0xF0)?;
                run = 0;
            }
            continue;
        }
        let s = category(v).min(10);
        let sym: u8 = (run << 4) | s;
        bw.put_huff(ac_ht, sym)?;
        let bits = amplitude_bits(v, s);
        bw.put_bits(bits, s)?;
        run = 0;
    }
    if run > 0 {
        bw.put_huff(ac_ht, 0x00)?;
    }
    Ok(())
}

// -------- Generators: grayscale and color --------

fn write_random_jpeg_gray(
    args: &Args,
    mut rng: impl Rng,
    mut w: impl Write,
) -> std::io::Result<()> {
    let mut bw = BitWriter::new(&mut w);

    // Tables
    let qy_nat = scaled_qtable_natural(&STD_LUMA_QTABLE_Q50_NATURAL, args.quality);
    let qy_zz = natural_to_zz(&qy_nat);

    let dc_y = HuffTable::from_spec(&STD_BITS_DC_LUMA, &STD_VALS_DC_LUMA);
    let ac_y = HuffTable::from_spec(&STD_BITS_AC_LUMA, &STD_VALS_AC_LUMA);

    // Headers
    write_soi(&mut bw)?;
    write_app0_jfif(&mut bw)?;
    write_dqt(&mut bw, &qy_zz, 0)?;
    write_sof0_gray(&mut bw, args.width, args.height, 0)?;
    write_dht_single(&mut bw, &STD_BITS_DC_LUMA, &STD_VALS_DC_LUMA, 0, 0)?;
    write_dht_single(&mut bw, &STD_BITS_AC_LUMA, &STD_VALS_AC_LUMA, 1, 0)?;
    write_sos_gray(&mut bw, 0, 0)?;

    // Scan
    let mcu_x = ((args.width as usize) + 7) / 8;
    let mcu_y = ((args.height as usize) + 7) / 8;

    // Prepare waves if needed
    let ac_indices = pick_active_indices(&mut rng, args.nonzero, args.freq, args.low_span);
    let mut s_bits_y = Vec::with_capacity(ac_indices.len());
    for _ in 0..ac_indices.len() {
        s_bits_y.push(rng.gen_range(1..=args.max_ac_bits));
    }
    let mut waves_y: Vec<Vec<Wave>> = Vec::with_capacity(ac_indices.len());
    if matches!(args.pattern, Pattern::Waves) {
        for _ in 0..ac_indices.len() {
            waves_y.push(make_waves(
                &mut rng,
                args.num_waves.max(1),
                mcu_x,
                mcu_y,
            ));
        }
    }

    let mut prev_dc_y: i32 = 0;
    let mut dc_bias_y: i32 = 0;

    for by in 0..mcu_y {
        for bx in 0..mcu_x {
            if args.dc_random_walk {
                dc_bias_y += rng.gen_range(-2..=2);
                dc_bias_y = dc_bias_y.clamp(-128, 127);
            } else {
                dc_bias_y = 0;
            }

            let block_y = match args.pattern {
                Pattern::Random => choose_random_block(
                    &mut rng,
                    args.nonzero,
                    args.max_ac_bits,
                    args.freq,
                    args.low_span,
                    dc_bias_y,
                ),
                Pattern::Waves => generate_block_waves(
                    &ac_indices,
                    &s_bits_y,
                    &waves_y,
                    bx,
                    by,
                    dc_bias_y,
                ),
            };

            encode_block(&mut bw, &dc_y, &ac_y, &mut prev_dc_y, &block_y)?;
        }
    }

    bw.flush_to_byte()?;
    write_eoi(&mut bw)?;
    Ok(())
}

fn write_random_jpeg_color(
    args: &Args,
    mut rng: impl Rng,
    mut w: impl Write,
) -> std::io::Result<()> {
    let mut bw = BitWriter::new(&mut w);

    // Quant tables
    let qy_nat = scaled_qtable_natural(&STD_LUMA_QTABLE_Q50_NATURAL, args.quality);
    let qc_nat = scaled_qtable_natural(&STD_CHROMA_QTABLE_Q50_NATURAL, args.quality);
    let qy_zz = natural_to_zz(&qy_nat);
    let qc_zz = natural_to_zz(&qc_nat);

    // Huffman tables (reuse luma tables for chroma)
    let dc_y = HuffTable::from_spec(&STD_BITS_DC_LUMA, &STD_VALS_DC_LUMA);
    let ac_y = HuffTable::from_spec(&STD_BITS_AC_LUMA, &STD_VALS_AC_LUMA);
    let dc_c = HuffTable::from_spec(&STD_BITS_DC_LUMA, &STD_VALS_DC_LUMA);
    let ac_c = HuffTable::from_spec(&STD_BITS_AC_LUMA, &STD_VALS_AC_LUMA);

    // Headers
    write_soi(&mut bw)?;
    write_app0_jfif(&mut bw)?;
    write_dqt(&mut bw, &qy_zz, 0)?; // Tq=0 for Y
    write_dqt(&mut bw, &qc_zz, 1)?; // Tq=1 for Cb/Cr
    write_sof0_ycc444(&mut bw, args.width, args.height, 0, 1)?;
    // DHTs: Y -> tid=0, C -> tid=1
    write_dht_single(&mut bw, &STD_BITS_DC_LUMA, &STD_VALS_DC_LUMA, 0, 0)?;
    write_dht_single(&mut bw, &STD_BITS_AC_LUMA, &STD_VALS_AC_LUMA, 1, 0)?;
    write_dht_single(&mut bw, &STD_BITS_DC_LUMA, &STD_VALS_DC_LUMA, 0, 1)?;
    write_dht_single(&mut bw, &STD_BITS_AC_LUMA, &STD_VALS_AC_LUMA, 1, 1)?;
    write_sos_ycc444(&mut bw)?;

    // MCU grid for 4:4:4 is 8x8 per component
    let mcu_x = ((args.width as usize) + 7) / 8;
    let mcu_y = ((args.height as usize) + 7) / 8;

    // Prepare active indices and per-channel waves if needed
    let ac_indices = pick_active_indices(&mut rng, args.nonzero, args.freq, args.low_span);

    // Per-index category selection per channel (keeps variety)
    let mut s_bits_y = Vec::with_capacity(ac_indices.len());
    let mut s_bits_cb = Vec::with_capacity(ac_indices.len());
    let mut s_bits_cr = Vec::with_capacity(ac_indices.len());
    for _ in 0..ac_indices.len() {
        s_bits_y.push(rng.gen_range(2..=args.max_ac_bits.max(2)));
        s_bits_cb.push(rng.gen_range(2..=args.max_ac_bits.max(2)));
        s_bits_cr.push(rng.gen_range(2..=args.max_ac_bits.max(2)));
    }

    let mut waves_y: Vec<Vec<Wave>> = Vec::new();
    let mut waves_cb: Vec<Vec<Wave>> = Vec::new();
    let mut waves_cr: Vec<Vec<Wave>> = Vec::new();
    if matches!(args.pattern, Pattern::Waves) {
        // Make separate wave sets per index and channel (colorful variation)
        for _ in 0..ac_indices.len() {
            waves_y.push(make_waves(
                &mut rng,
                args.num_waves.max(1),
                mcu_x,
                mcu_y,
            ));
            waves_cb.push(make_waves(
                &mut rng,
                args.num_waves.max(1),
                mcu_x,
                mcu_y,
            ));
            waves_cr.push(make_waves(
                &mut rng,
                args.num_waves.max(1),
                mcu_x,
                mcu_y,
            ));
        }
    }

    let mut prev_dc_y: i32 = 0;
    let mut prev_dc_cb: i32 = 0;
    let mut prev_dc_cr: i32 = 0;

    let mut dc_bias_y: i32 = 0;
    let mut dc_bias_cb: i32 = 0;
    let mut dc_bias_cr: i32 = 0;

    for by in 0..mcu_y {
        for bx in 0..mcu_x {
            if args.dc_random_walk {
                dc_bias_y += rng.gen_range(-2..=2);
                dc_bias_cb += rng.gen_range(-2..=2);
                dc_bias_cr += rng.gen_range(-2..=2);
                dc_bias_y = dc_bias_y.clamp(-128, 127);
                dc_bias_cb = dc_bias_cb.clamp(-128, 127);
                dc_bias_cr = dc_bias_cr.clamp(-128, 127);
            } else {
                dc_bias_y = 0;
                dc_bias_cb = 0;
                dc_bias_cr = 0;
            }

            let block_y = match args.pattern {
                Pattern::Random => choose_random_block(
                    &mut rng,
                    args.nonzero,
                    args.max_ac_bits,
                    args.freq,
                    args.low_span,
                    dc_bias_y,
                ),
                Pattern::Waves => generate_block_waves(
                    &ac_indices,
                    &s_bits_y,
                    &waves_y,
                    bx,
                    by,
                    dc_bias_y,
                ),
            };
            encode_block(&mut bw, &dc_y, &ac_y, &mut prev_dc_y, &block_y)?;

            let block_cb = match args.pattern {
                Pattern::Random => choose_random_block(
                    &mut rng,
                    args.nonzero,
                    args.max_ac_bits,
                    args.freq,
                    args.low_span,
                    dc_bias_cb,
                ),
                Pattern::Waves => generate_block_waves(
                    &ac_indices,
                    &s_bits_cb,
                    &waves_cb,
                    bx,
                    by,
                    dc_bias_cb,
                ),
            };
            encode_block(&mut bw, &dc_c, &ac_c, &mut prev_dc_cb, &block_cb)?;

            let block_cr = match args.pattern {
                Pattern::Random => choose_random_block(
                    &mut rng,
                    args.nonzero,
                    args.max_ac_bits,
                    args.freq,
                    args.low_span,
                    dc_bias_cr,
                ),
                Pattern::Waves => generate_block_waves(
                    &ac_indices,
                    &s_bits_cr,
                    &waves_cr,
                    bx,
                    by,
                    dc_bias_cr,
                ),
            };
            encode_block(&mut bw, &dc_c, &ac_c, &mut prev_dc_cr, &block_cr)?;
        }
    }

    bw.flush_to_byte()?;
    write_eoi(&mut bw)?;
    Ok(())
}

fn choose_random_block(
    rng: &mut impl Rng,
    nonzero: usize,
    max_ac_bits: u8,
    freq: FreqMode,
    low_span: usize,
    dc_val: i32,
) -> [i16; 64] {
    let mut block_zz = [0i16; 64];
    block_zz[0] = dc_val as i16;

    let max_bits = max_ac_bits.clamp(1, 10);
    let mut indices: Vec<usize> = match freq {
        FreqMode::Low => {
            let span = low_span.clamp(1, 63);
            (1..=span).collect()
        }
        FreqMode::All => (1..64).collect(),
    };
    indices.shuffle(rng);
    let take = nonzero.min(indices.len());
    for &i in indices.iter().take(take) {
        let s = rng.gen_range(1..=max_bits);
        let max_mag = (1i32 << s) - 1;
        let mag = rng.gen_range(1..=max_mag) as i32;
        let sign = if rng.gen_bool(0.5) { 1 } else { -1 };
        let v = (sign * mag) as i16;
        block_zz[i] = v;
    }
    block_zz
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    if args.width < 8 || args.height < 8 {
        eprintln!("Width and height must be at least 8.");
        std::process::exit(1);
    }
    if args.max_ac_bits < 1 || args.max_ac_bits > 10 {
        eprintln!("--max-ac-bits must be in 1..=10");
        std::process::exit(1);
    }
    if args.nonzero > 63 {
        eprintln!("--nonzero must be in 0..=63");
        std::process::exit(1);
    }
    if args.freq == FreqMode::Low && (args.low_span < 1 || args.low_span > 63) {
        eprintln!("--low-span must be in 1..=63");
        std::process::exit(1);
    }
    if args.quality == 0 || args.quality > 100 {
        eprintln!("--quality must be in 1..=100");
        std::process::exit(1);
    }

    let mut rng: StdRng = match args.seed {
        Some(s) => SeedableRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let f = File::create(&args.output)?;
    let mut writer = BufWriter::new(f);

    if args.grayscale {
        write_random_jpeg_gray(&args, &mut rng, &mut writer)?;
    } else {
        write_random_jpeg_color(&args, &mut rng, &mut writer)?;
    }
    writer.flush()?;
    Ok(())
}