use ::neural_network as nn;
use nn::{
    layers::{convolution::Padding, Layer},
    NeuralNetwork,
};
use rand::{CryptoRng, RngCore};

use super::*;

pub fn construct_mnist<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    num_poly: usize,
    rng: &mut R,
) -> NeuralNetwork<TenBitAS, TenBitExpFP> {
    let relu_layers = match num_poly {
        0 => vec![1, 4, 7],
        1 => vec![1, 4],
        2 => vec![1],
        3 => vec![],
        _ => unreachable!(),
    };

    let mut network = match &vs {
        Some(vs) => NeuralNetwork {
            layers: vec![],
            eval_method: ::neural_network::EvalMethod::TorchDevice(vs.device()),
        },
        None => NeuralNetwork {
            layers: vec![],
            ..Default::default()
        },
    };
    // Dimensions of input image.
    let input_dims = (batch_size, 1, 28, 28);

    let kernel_dims = (16, 1, 5, 5);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);

    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    network.layers.push(Layer::LL(pool));

    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (16, 16, 5, 5);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);

    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    network.layers.push(Layer::LL(pool));

    let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    println!("Fc input dims: {:?}", fc_input_dims);
    let (fc, _) = sample_fc_layer(vs, fc_input_dims, 100, rng);
    network.layers.push(Layer::LL(fc));
    add_activation_layer(&mut network, &relu_layers);

    let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    println!("Fc input dims: {:?}", fc_input_dims);
    let (fc, _) = sample_fc_layer(vs, fc_input_dims, 10, rng);
    network.layers.push(Layer::LL(fc));

    for layer in &network.layers {
        println!("Layer dim: {:?}", layer.input_dimensions());
    }

    assert!(network.validate());

    network
}
