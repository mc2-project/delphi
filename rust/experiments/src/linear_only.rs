use ::neural_network as nn;
use nn::{
    layers::{convolution::Padding, Layer},
    NeuralNetwork,
};
use rand::{CryptoRng, RngCore};

type InputDims = (usize, usize, usize, usize);

use super::*;

pub fn construct_networks<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    rng: &mut R,
) -> Vec<(InputDims, NeuralNetwork<TenBitAS, TenBitExpFP>)> {
    let mut networks = Vec::new();
    let input_dims = [
        (batch_size, 3, 32, 32),
        (batch_size, 16, 32, 32),
        (batch_size, 32, 16, 16),
        (batch_size, 64, 8, 8),
    ];
    let kernel_dims = [
        (16, 3, 3, 3),
        (16, 16, 3, 3),
        (32, 32, 3, 3),
        (64, 64, 3, 3),
    ];

    for i in 0..4 {
        let input_dims = input_dims[i];
        let kernel_dims = kernel_dims[i];
        let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
        let network = match &vs {
            Some(vs) => NeuralNetwork {
                layers:      vec![Layer::LL(conv)],
                eval_method: ::neural_network::EvalMethod::TorchDevice(vs.device()),
            },
            None => NeuralNetwork {
                layers: vec![Layer::LL(conv)],
                ..Default::default()
            },
        };
        networks.push((input_dims, network));
    }
    networks
}
