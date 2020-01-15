use crate::*;
use protocols::neural_network::NNProtocol;
use rand::{CryptoRng, RngCore};
use std::{io::BufReader, net::TcpListener};

pub fn nn_server<R: RngCore + CryptoRng>(
    server_addr: &str,
    nn: NeuralNetwork<TenBitAS, TenBitExpFP>,
    rng: &mut R,
) {
    let server_listener = TcpListener::bind(server_addr).unwrap();
    let server_offline_state = {
        let stream = server_listener
            .incoming()
            .next()
            .unwrap()
            .expect("server connection failed!");
        let write_stream = stream.try_clone().unwrap();
        let read_stream = BufReader::new(stream);
        NNProtocol::offline_server_protocol(read_stream, write_stream, &nn, rng).unwrap()
    };

    let _ = {
        let stream = server_listener
            .incoming()
            .next()
            .unwrap()
            .expect("server connection failed!");
        let write_stream = stream.try_clone().unwrap();
        let read_stream = BufReader::new(stream);
        NNProtocol::online_server_protocol(read_stream, write_stream, &nn, &server_offline_state)
            .unwrap()
    };
}
