use super::*;
use crate::*;
use algebra::fields::near_mersenne_64::F;
use rand::Rng;
use rand_chacha::ChaChaRng;

struct TenBitExpParams {}
impl FixedPointParameters for TenBitExpParams {
    type Field = F;
    const MANTISSA_CAPACITY: u8 = 3;
    const EXPONENT_CAPACITY: u8 = 5;
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

mod convolution {
    use super::*;
    use rand::SeedableRng;
    use tch::nn;

    #[test]
    fn test_torch() {
        println!("CUDA device is available: {:?}", tch::Cuda::is_available());
        let vs = nn::VarStore::new(tch::Device::cuda_if_available());
        println!("VS Device: {:?}", vs.device());
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        // Set the parameters for the convolution.
        let input_dims = (1, 1, 3, 3);
        let kernel_dims = (1, 1, 3, 3);
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

        let naive_method = crate::EvalMethod::Naive;
        let torch_method = crate::EvalMethod::TorchDevice(vs.device());
        let result_1 = layer.evaluate_with_method(torch_method, &input);
        let result_2 = layer.evaluate_with_method(naive_method, &input);

        println!("Naive result: [");
        for result in &result_2 {
            println!("  {},", result);
        }
        println!("]");

        println!("Torch result: [");
        for result in &result_1 {
            println!("  {},", result);
        }
        println!("]");

        assert_eq!(result_1, result_2);
    }

    #[test]
    fn load_conv2d_weights() {
        let input: Input<TenBitExpFP> = ndarray::Array4::from_shape_vec(
            (1, 1, 3, 3),
            vec![
                1.0, 1.0, 1.0, //
                1.0, 1.0, 0.0, //
                1.0, 1.0, 1.0, //
            ],
        )
        .expect("Should be of correct shape")
        .into();

        let fp_kernel = ndarray::Array4::from_shape_vec(
            (1, 1, 3, 3),
            vec![
                1.0, 1.0, 1.0, //
                0.0, 0.0, 0.0, //
                1.0, 1.0, 1.0, //
            ],
        )
        .expect("Should be of correct shape");

        let kernel: Kernel<TenBitExpFP> = fp_kernel.into();
        let mut bias: Kernel<TenBitExpFP> = Kernel::zeros((1, 1, 1, 1));
        bias.iter_mut().for_each(|ker_i| {
            *ker_i = TenBitExpFP::from(1.0);
        });

        let conv_params = Conv2dParams::<TenBitExpFP, _>::new(Padding::Same, 1, kernel, bias);
        let mut output = Output::zeros((1, 1, 3, 3));
        conv_params.conv2d_naive(&input, &mut output);
        output.iter_mut().for_each(|e| {
            e.signed_reduce_in_place();
        });

        let expected_output: Output<TenBitExpFP> = ndarray::Array4::from_shape_vec(
            (1, 1, 3, 3),
            vec![
                3.0, 3.0, 2.0, //
                5.0, 7.0, 5.0, //
                3.0, 3.0, 2.0,
            ],
        )
        .unwrap()
        .into();
        output
            .iter()
            .for_each(|e| println!("e: {:?}", e.signed_reduce()));
        expected_output.iter().for_each(|e| println!("e: {:?}", e));
        assert_eq!(output, expected_output);
    }

    #[test]
    fn check_naive_conv2d() {
        let input: Input<TenBitExpFP> = ndarray::Array4::from_shape_vec(
            (1, 3, 5, 5),
            vec![
                0.0, 0.0, 0.0, 0.0, 1.0f64, //
                1.0, 2.0, 0.0, 1.0, 0.0, //
                0.0, 1.0, 2.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, 2.0, 1.0, //
                1.0, 2.0, 1.0, 0.0, 0.0, //
                //
                1.0, 1.0, 1.0, 1.0, 1.0, //
                2.0, 2.0, 2.0, 1.0, 0.0, //
                1.0, 0.0, 0.0, 0.0, 2.0, //
                0.0, 1.0, 2.0, 1.0, 0.0, //
                1.0, 1.0, 2.0, 0.0, 1.0, //
                //
                0.0, 0.0, 2.0, 2.0, 2.0, //
                1.0, 2.0, 0.0, 1.0, 2.0, //
                1.0, 0.0, 2.0, 2.0, 1.0, //
                0.0, 2.0, 0.0, 1.0, 0.0, //
                2.0, 1.0, 1.0, 1.0, 0.0, //
            ],
        )
        .expect("Should be of correct shape")
        .into();

        let fp_kernel = ndarray::Array4::from_shape_vec(
            (1, 3, 3, 3),
            vec![
                -1.0, 1.0, 0.0f64, //
                0.0, 0.0, 0.0, //
                -1.0, 0.0, 0.0, //
                //
                1.0, -1.0, -1.0, //
                -1.0, 0.0, 1.0, //
                1.0, 1.0, -1.0, //
                //
                -1.0, -1.0, 1.0, //
                0.0, 1.0, 0.0, //
                1.0, -1.0, 1.0, //
            ],
        )
        .expect("Should be of correct shape");

        let kernel: Kernel<TenBitExpFP> = fp_kernel.into();
        let mut bias: Kernel<TenBitExpFP> = Kernel::zeros((1, 1, 1, 1));
        bias.iter_mut().for_each(|ker_i| {
            *ker_i = TenBitExpFP::from(1.0);
        });

        let conv_params =
            Conv2dParams::<TenBitExpFP, _>::new(Padding::Same, 2, kernel.clone(), bias.clone());
        let mut output = Output::zeros((1, 1, 3, 3));
        conv_params.conv2d_naive(&input, &mut output);

        let mut expected_output: Output<TenBitExpFP> = ndarray::Array4::from_shape_vec(
            (1, 1, 3, 3),
            vec![
                2.0, 6.0, 0.0f64, //
                0.0, 3.0, -2.0, //
                5.0, -3.0, -1.0, //
            ],
        )
        .unwrap()
        .into();
        expected_output
            .iter_mut()
            .for_each(|e| *e += TenBitExpFP::from(1.0));
        assert_eq!(output, expected_output);

        let conv_params =
            Conv2dParams::<TenBitExpFP, _>::new(Padding::Valid, 1, kernel.clone(), bias.clone());
        output = Output::zeros((1, 1, 3, 3));
        conv_params.conv2d_naive(&input, &mut output);

        expected_output = ndarray::Array4::from_shape_vec(
            (1, 1, 3, 3),
            vec![
                7.0, -3.0, -7.0f64, //
                -9.0, 3.0, 9.0, //
                8.0, 3.0, -8.0, //
            ],
        )
        .unwrap()
        .into();
        expected_output
            .iter_mut()
            .for_each(|e| *e += TenBitExpFP::from(1.0));
        assert_eq!(output, expected_output);

        let input: Input<TenBitExpFP> = ndarray::Array4::from_shape_vec(
            (1, 1, 3, 3),
            vec![
                1.0, 2.0, 1.0, //
                2.0, 1.0, 2.0, //
                1.0, 1.0, 1.0, //
            ],
        )
        .expect("Should be of correct shape")
        .into();

        let fp_kernel = ndarray::Array4::from_shape_vec((3, 1, 1, 1), vec![1.0, 2.0, 3.0])
            .expect("Should be of correct shape");

        let fp_bias = ndarray::Array4::from_shape_vec((3, 1, 1, 1), vec![7.0, 3.0, 9.0])
            .expect("Should be of correct shape");

        let kernel: Kernel<TenBitExpFP> = fp_kernel.into();
        let bias: Kernel<TenBitExpFP> = fp_bias.into();

        let conv_params =
            Conv2dParams::<TenBitExpFP, _>::new(Padding::Valid, 1, kernel.clone(), bias.clone());
        let mut output = Output::zeros((1, 3, 3, 3));
        conv_params.conv2d_naive(&input, &mut output);

        expected_output = ndarray::Array4::from_shape_vec(
            (1, 3, 3, 3),
            vec![
                1.0, 2.0, 1.0f64, //
                2.0, 1.0, 2.0, //
                1.0, 1.0, 1.0, //
                //
                2.0, 4.0, 2.0f64, //
                4.0, 2.0, 4.0, //
                2.0, 2.0, 2.0, //
                //
                3.0, 6.0, 3.0f64, //
                6.0, 3.0, 6.0, //
                3.0, 3.0, 3.0, //
            ],
        )
        .unwrap()
        .into();

        expected_output
            .slice_mut(s![0, .., .., ..])
            .outer_iter_mut()
            .enumerate()
            .for_each(|(i, mut view)| {
                view.iter_mut().for_each(|e| *e += bias[[i, 0, 0, 0]]);
            });
        assert_eq!(output, expected_output);
    }
}

mod pooling {
    use super::*;
    #[test]
    fn check_naive_avg_pool() {
        let input: Input<TenBitExpFP> = ndarray::Array4::from_shape_vec(
            (1, 2, 4, 4),
            vec![
                0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 2.0,
                1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0,
            ],
        )
        .unwrap()
        .into();
        let mut output = Output::zeros((1, 2, 2, 2));

        let normalizer = TenBitExpFP::from(1.0 / (2.0 * 2.0));
        let pool_params = AvgPoolParams::<TenBitExpFP, _>::new(2, 2, 2, normalizer);
        pool_params.avg_pool_naive(&input, &mut output);

        let expected_output = ndarray::Array4::from_shape_vec(
            (1, 2, 2, 2),
            vec![0.75, 0.25, 0.5, 1.0, 1.5, 1.25, 0.5, 0.75],
        )
        .unwrap()
        .into();
        assert_eq!(output, expected_output);
    }
}
