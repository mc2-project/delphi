use criterion::{criterion_group, criterion_main, Criterion};
// use itertools::Itertools;

use std::time::Duration;

use algebra::{fields::near_mersenne_64::F, fixed_point::*};
use crypto_primitives::{additive_share::Share, beavers_mul::*};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;

fn generate_random_number<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
    let is_neg: bool = rng.gen();
    let mul = if is_neg { -10.0 } else { 10.0 };
    let float: f64 = rng.gen();
    let f = TenBitExpFP::truncate_float(float * mul);
    let n = TenBitExpFP::from(f);
    (f, n)
}

struct TenBitExpParams {}
impl FixedPointParameters for TenBitExpParams {
    type Field = F;
    const MANTISSA_CAPACITY: u8 = 5;
    const EXPONENT_CAPACITY: u8 = 5;
}

type TenBitExpFP = FixedPoint<TenBitExpParams>;

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

fn bench_beavers_mul(c: &mut Criterion) {
    c.bench_function_over_inputs(
        &format!("beavers_mul"),
        move |bench, num| {
            let mut rng = ChaChaRng::from_seed(RANDOMNESS);
            let seed = RANDOMNESS;
            let mut gen = InsecureTripleGen::<TenBitExpFP>::new(seed);
            let mut inputs = vec![];
            for _ in 0..1000 {
                let (t1, t2) = gen.generate_triple_shares();
                let (_, n1) = generate_random_number(&mut rng);
                let (_, n2) = generate_random_number(&mut rng);

                let (s11, s12) = n1.share(&mut rng);
                let (s21, s22) = n2.share(&mut rng);

                inputs.push([(s11, s12, t1), (s21, s22, t2)]);
            }

            let mut i = 0;
            bench.iter(|| {
                for _ in 0..*num {
                    let [(s11, s12, t1), (s21, s22, t2)] = inputs[i % 1000].clone();
                    let p1_bl_input = FPBeaversMul::share_and_blind_inputs(&s11, &s21, &t1);
                    let p2_bl_input = FPBeaversMul::share_and_blind_inputs(&s12, &s22, &t2);
                    let (p1_bl_input, p2_bl_input) = (
                        FPBeaversMul::reconstruct_blinded_inputs(p1_bl_input, p2_bl_input),
                        FPBeaversMul::reconstruct_blinded_inputs(p2_bl_input, p1_bl_input),
                    );

                    let mut s31 = FPBeaversMul::multiply_blinded_inputs(1, p1_bl_input, &t1);
                    let mut s32 = FPBeaversMul::multiply_blinded_inputs(2, p2_bl_input, &t2);
                    s31.inner.signed_reduce_in_place();
                    s32.inner.signed_reduce_in_place();
                    let n4 = s31.combine(&s32);
                    i += 1;
                    let _ = criterion::black_box(n4);
                }
            });
        },
        vec![1, 10, 100, 1000, 10000],
    );
}

criterion_group! {
    name = beavers_mul;
    config = Criterion::default().warm_up_time(Duration::from_millis(100));
    targets = bench_beavers_mul
}

criterion_main!(beavers_mul);
