use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

use algebra::{fields::near_mersenne_64::F, fixed_point::*};
use neural_network::{layers::average_pooling::*, tensors::*};
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

fn make_avg_pool(
    c: &mut Criterion,
    input_dim: (usize, usize, usize, usize),
    pool_h: usize,
    pool_w: usize,
    stride: usize,
) {
    c.bench_function(
        &format!("Avg pooling {:?} ({:?}x{:?})", input_dim, pool_h, pool_w),
        move |bench| {
            let mut rng = ChaChaRng::from_seed(RANDOMNESS);
            let mut input = Input::zeros(input_dim);
            for in_i in input.iter_mut() {
                let (_, n) = generate_random_number(&mut rng);
                *in_i = n;
            }
            let normalizer = TenBitExpFP::from(1.0 / (2.0 * 2.0));
            let pool_params =
                AvgPoolParams::<TenBitExpFP, _>::new(pool_h, pool_w, stride, normalizer);
            let output_dims = pool_params.calculate_output_size(input.dim());
            let mut output = Input::zeros(output_dims);
            bench.iter(|| pool_params.avg_pool_naive(&input, &mut output));
        },
    );
}

fn bench_avg_pool_128x16x16_by_2x2(c: &mut Criterion) {
    let input_dim = (1, 128, 16, 16);
    let stride = 2;
    make_avg_pool(c, input_dim, 2, 2, stride);
}

fn bench_avg_pool_1x28x28_by_2x2(c: &mut Criterion) {
    let input_dim = (1, 1, 28, 28);
    let stride = 2;
    make_avg_pool(c, input_dim, 2, 2, stride);
}

criterion_group! {
    name = pooling;
    config = Criterion::default().warm_up_time(Duration::from_millis(100));
    targets = bench_avg_pool_128x16x16_by_2x2, bench_avg_pool_1x28x28_by_2x2
}

criterion_main!(pooling);
