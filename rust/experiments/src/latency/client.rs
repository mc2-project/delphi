use crate::*;
use ::neural_network::{tensors::Input, NeuralArchitecture};
use protocols::neural_network::NNProtocol;
use std::{io::BufReader, net::TcpStream};

pub fn nn_client<R: RngCore + CryptoRng>(
    server_addr: &str,
    architecture: NeuralArchitecture<TenBitAS, TenBitExpFP>,
    rng: &mut R,
) {
    // Sample a random input.
    let input_dims = architecture.layers.first().unwrap().input_dimensions();
    let mut input = Input::zeros(input_dims);
    input
        .iter_mut()
        .for_each(|in_i| *in_i = generate_random_number(rng).1);

    let client_state = {
        // client's connection to server.
        let stream = TcpStream::connect(server_addr).expect("connecting to server failed");
        let read_stream = BufReader::new(stream.try_clone().unwrap());
        let write_stream = stream;

        NNProtocol::offline_client_protocol(read_stream, write_stream, &architecture, rng).unwrap()
    };

    let _client_output = {
        let stream = TcpStream::connect(server_addr).expect("connecting to server failed");
        let read_stream = BufReader::new(stream.try_clone().unwrap());
        let write_stream = stream;
        NNProtocol::online_client_protocol(
            read_stream,
            write_stream,
            &input,
            &architecture,
            &client_state,
        )
        .unwrap()
    };
}
