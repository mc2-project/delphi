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
            let mut read_stream = IMuxSync::new(vec![BufReader::new(stream.try_clone().unwrap())]);
            let mut write_stream = IMuxSync::new(vec![stream]);

            NNProtocol::offline_server_protocol(&mut read_stream, &mut write_stream, &nn, rng)
                .unwrap()
        };
        server_states.push(server_state);
    }

    let _ = crossbeam::thread::scope(|s| {
        let mut results = Vec::new();
        for stream in server_listener.incoming() {
            let result = s.spawn(|_| {
                let stream = stream.expect("server connection failed!");
                let mut read_stream =
                    IMuxSync::new(vec![BufReader::new(stream.try_clone().unwrap())]);
                let mut write_stream = IMuxSync::new(vec![stream]);
                NNProtocol::online_server_protocol(
                    &mut read_stream,
                    &mut write_stream,
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
