#![feature(test)]
extern crate test;
use algebra::{fields::near_mersenne_64::F, *};
use neural_network::{
    layers::{convolution::*, *},
    tensors::*,
};
use rand::Rng;
use rand_chacha::ChaChaRng;
use test::Bencher;

struct TenBitExpParams {}
impl FixedPointParameters for TenBitExpParams {
    type Field = F;
    const MANTISSA_CAPACITY: u8 = 8;
    const EXPONENT_CAPACITY: u8 = 6;
}

type TenBitExpFP = FixedPoint<TenBitExpParams>;

pub const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x84, 0xbc, 0x89, 0xa7, 0x94, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x43, 0x72, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd1,
];

fn generate_random_number<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
    let is_neg: bool = rng.gen();
    let mul = if is_neg { -10.0 } else { 10.0 };
    let float: f64 = rng.gen();
    let f = TenBitExpFP::truncate_float(float * mul);
    let n = TenBitExpFP::from(f);
    (f, n)
}
use neural_network::*;
use rand::SeedableRng;
use tch::nn;

#[bench]
fn test_torch_cpu(b: &mut Bencher) {
    println!("CUDA device is available: {:?}", tch::Cuda::is_available());
    let vs = nn::VarStore::new(tch::Device::cuda_if_available());
    println!("VS Device: {:?}", vs.device());
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    // Set the parameters for the convolution.
    let input_dims = (10, 16, 32, 32);
    let kernel_dims = (16, 16, 3, 3);
    let stride = 1;
    let padding = Padding::Same;
    // Sample a random kernel.
    let mut kernel = Kernel::zeros(kernel_dims);
    let mut bias = Kernel::zeros((kernel_dims.0, 1, 1, 1));
    kernel
        .iter_mut()
        .for_each(|ker_i| *ker_i = generate_random_number(&mut rng).1);
    bias.iter_mut()
        .for_each(|bias_i| *bias_i = generate_random_number(&mut rng).1);

    let conv_params =
        Conv2dParams::<TenBitExpFP, _>::new_with_gpu(&vs.root(), padding, stride, kernel, bias);
    let output_dims = conv_params.calculate_output_size(input_dims);
    let layer_dims = LayerDims {
        input_dims,
        output_dims,
    };
    let layer = LinearLayer::Conv2d {
        dims: layer_dims,
        params: conv_params,
    };

    let mut input = Input::zeros(input_dims);
    input
        .iter_mut()
        .for_each(|in_i| *in_i = generate_random_number(&mut rng).1);

    let naive_method = neural_network::EvalMethod::Naive;
    b.iter(|| layer.evaluate_with_method(naive_method, &input))
}

#[bench]
fn test_torch_gpu(b: &mut Bencher) {
    println!("CUDA device is available: {:?}", tch::Cuda::is_available());
    let vs = nn::VarStore::new(tch::Device::cuda_if_available());
    println!("VS Device: {:?}", vs.device());
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    // Set the parameters for the convolution.
    let input_dims = (10, 16, 32, 32);
    let kernel_dims = (16, 16, 3, 3);
    let stride = 1;
    let padding = Padding::Same;
    // Sample a random kernel.
    let mut kernel = Kernel::zeros(kernel_dims);
    let mut bias = Kernel::zeros((kernel_dims.0, 1, 1, 1));
    kernel
        .iter_mut()
        .for_each(|ker_i| *ker_i = generate_random_number(&mut rng).1);
    bias.iter_mut()
        .for_each(|bias_i| *bias_i = generate_random_number(&mut rng).1);

    let conv_params =
        Conv2dParams::<TenBitExpFP, _>::new_with_gpu(&vs.root(), padding, stride, kernel, bias);
    let output_dims = conv_params.calculate_output_size(input_dims);
    let layer_dims = LayerDims {
        input_dims,
        output_dims,
    };
    let layer = LinearLayer::Conv2d {
        dims: layer_dims,
        params: conv_params,
    };

    let mut input = Input::zeros(input_dims);
    input
        .iter_mut()
        .for_each(|in_i| *in_i = generate_random_number(&mut rng).1);

    let torch_method = neural_network::EvalMethod::TorchDevice(vs.device());
    b.iter(|| layer.evaluate_with_method(torch_method, &input))
}
