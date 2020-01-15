use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

use algebra::{fields::near_mersenne_64::F, fixed_point::*};
use neural_network::{layers::convolution::*, tensors::*};
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

fn make_conv2d(
    c: &mut Criterion,
    input_dim: (usize, usize, usize, usize),
    kernel_dim: (usize, usize, usize, usize),
    stride: usize,
    padding: Padding,
) {
    c.bench_function(
        &format!("convolution_{:?}_{:?}", input_dim, kernel_dim),
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
            let conv_params = Conv2dParams::<TenBitExpFP, _>::new(padding, stride, kernel, bias);
            let output_dims = conv_params.calculate_output_size(input_dim);

            let mut output = Output::zeros(output_dims);
            bench.iter(|| conv_params.conv2d_naive(&input, &mut output));
        },
    );
}

fn bench_conv2d_resnet32_2_1(c: &mut Criterion) {
    let input_dim = (1, 32, 16, 16);
    let kernel_dim = (32, 32, 3, 3);
    let stride = 1;
    let padding = Padding::Same;
    make_conv2d(c, input_dim, kernel_dim, stride, padding);
}

fn bench_conv2d_resnet32_3_3(c: &mut Criterion) {
    let input_dim = (1, 64, 8, 8);
    let kernel_dim = (64, 64, 3, 3);
    let stride = 1;
    let padding = Padding::Same;
    make_conv2d(c, input_dim, kernel_dim, stride, padding);
}

fn bench_conv2d_resnet32_1_1(c: &mut Criterion) {
    let input_dim = (1, 16, 32, 32);
    let kernel_dim = (16, 16, 3, 3);
    let stride = 1;
    let padding = Padding::Same;
    make_conv2d(c, input_dim, kernel_dim, stride, padding);
}

criterion_group! {
    name = convolution;
    config = Criterion::default().warm_up_time(Duration::from_millis(100));
    targets = bench_conv2d_resnet32_3_3,
              bench_conv2d_resnet32_1_1,
              bench_conv2d_resnet32_2_1
}

criterion_main!(convolution);
