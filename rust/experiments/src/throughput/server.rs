use crate::*;
use protocols::neural_network::NNProtocol;
use rand::{CryptoRng, RngCore};
use std::{io::BufReader, net::TcpListener};

pub fn nn_server<R: RngCore + CryptoRng>(
    server_addr: &str,
    nns: &[(
        (usize, usize, usize, usize),
        NeuralNetwork<TenBitAS, TenBitExpFP>,
    )],
    rng: &mut R,
) {
    let server_listener = TcpListener::bind(server_addr).unwrap();

    let mut server_states = Vec::new();
    for (_, nn) in nns {
        let server_state = {
            // client's connection to server.
            let stream = server_listener
                .incoming()
                .next()
                .unwrap()
                .expect("server connection failed!");
            let write_stream = stream.try_clone().unwrap();
            let read_stream = BufReader::new(stream);

            NNProtocol::offline_server_protocol(read_stream, write_stream, &nn, rng).unwrap()
        };
        server_states.push(server_state);
    }

    let _ = crossbeam::thread::scope(|s| {
        let mut results = Vec::new();
        for stream in server_listener.incoming() {
            let result = s.spawn(|_| {
                let read_stream = stream.expect("server connection failed!");
                let write_stream = read_stream.try_clone().unwrap();
                let read_stream = BufReader::new(read_stream);
                NNProtocol::online_server_protocol(
                    read_stream,
                    write_stream,
                    &nns[0].1,
                    &server_states[0],
                )
                .unwrap()
            });
            results.push(result);
        }
        for result in results {
            let _ = result.join().unwrap();
        }
    })
    .unwrap();
}
