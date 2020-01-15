use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

use algebra::{fields::near_mersenne_64::F, fixed_point::*};
use neural_network::{layers::fully_connected::*, tensors::*};
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

fn make_fc(
    c: &mut Criterion,
    input_dim: (usize, usize, usize, usize),
    kernel_dim: (usize, usize, usize, usize),
) {
    c.bench_function(
        &format!("fully_connected_{:?}_{:?}", input_dim, kernel_dim),
        move |bench| {
            let mut rng = ChaChaRng::from_seed(RANDOMNESS);
            let mut input = Input::zeros(input_dim);
            for in_i in input.iter_mut() {
                let (_, n) = generate_random_number(&mut rng);
                *in_i = n;
            }
            let mut kernel = Kernel::zeros(kernel_dim);
            for ker_i in kernel.iter_mut() {
                let (_, n) = generate_random_number(&mut rng);
                *ker_i = n;
            }
            let mut bias = Kernel::zeros((kernel_dim.0, 1, 1, 1));
            for b_i in bias.iter_mut() {
                let (_, n) = generate_random_number(&mut rng);
                *b_i = n;
            }
            let fc_params = FullyConnectedParams::<TenBitExpFP, _>::new(kernel, bias);

            let output_dims = fc_params.calculate_output_size(input_dim);
            let mut output = Input::zeros(output_dims);
            bench.iter(|| fc_params.fully_connected_naive(&input, &mut output));
        },
    );
}

fn bench_fc_16x8x8_to_100x1x1(c: &mut Criterion) {
    let input_dim = (1, 16, 8, 8);
    let kernel_dim = (100, 16, 8, 8);
    make_fc(c, input_dim, kernel_dim);
}

fn bench_fc_32x16x16_to_4096x1x1(c: &mut Criterion) {
    let input_dim = (1, 32, 16, 16);
    let kernel_dim = (4096, 32, 16, 16);
    make_fc(c, input_dim, kernel_dim);
}
criterion_group! {
    name = fully_connected;
    config = Criterion::default().warm_up_time(Duration::from_millis(100));
    targets = bench_fc_16x8x8_to_100x1x1
}

criterion_group! {
    name = fully_connected_slow;
    config = Criterion::default().warm_up_time(Duration::from_millis(100)).sample_size(2);
    targets = bench_fc_32x16x16_to_4096x1x1
}

criterion_main!(fully_connected, fully_connected_slow);
