use ::neural_network as nn;
use nn::{
    layers::{convolution::Padding, Layer},
    NeuralNetwork,
};
use rand::{CryptoRng, RngCore};

use super::*;

pub fn construct_minionn<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    num_poly: usize,
    rng: &mut R,
) -> NeuralNetwork<TenBitAS, TenBitExpFP> {
    let relu_layers = match num_poly {
        0 => vec![1, 3, 6, 8, 11, 13, 15],
        1 => vec![1, 3, 6, 8, 11, 13],
        2 => vec![1, 3, 6, 8, 11],
        3 => vec![3, 11, 13, 15],
        5 => vec![6, 11],
        6 => vec![11],
        7 => vec![],
        _ => unreachable!(),
    };

    let mut network = match &vs {
        Some(vs) => NeuralNetwork {
            layers:      vec![],
            eval_method: ::neural_network::EvalMethod::TorchDevice(vs.device()),
        },
        None => NeuralNetwork {
            layers: vec![],
            ..Default::default()
        },
    };
    // Dimensions of input image.
    let input_dims = (batch_size, 3, 32, 32);

    // 1
    let kernel_dims = (64, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // 2
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // 3
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    network.layers.push(Layer::LL(pool));
    // 4
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // 5
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // 6
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    network.layers.push(Layer::LL(pool));
    // 7
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // 8
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 1, 1);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // 9
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (16, 64, 1, 1);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    // 10
    let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    let (fc, _) = sample_fc_layer(vs, fc_input_dims, 10, rng);
    network.layers.push(Layer::LL(fc));
    assert!(network.validate());

    network
}
