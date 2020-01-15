use crate::*;
use neural_network::{ndarray::Array4, tensors::Input, NeuralArchitecture};
use protocols::neural_network::NNProtocol;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use std::{
    cmp,
    io::BufReader,
    net::{TcpListener, TcpStream},
};

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

pub fn softmax(x: &Input<TenBitExpFP>) -> Input<TenBitExpFP> {
    let mut max: TenBitExpFP = x[[0, 0, 0, 0]];
    x.iter().for_each(|e| {
        max = match max.cmp(e) {
            cmp::Ordering::Less => *e,
            _ => max,
        };
    });
    let mut e_x: Input<TenBitExpFP> = x.clone();
    e_x.iter_mut().for_each(|e| {
        *e = f64::from(*e - max).exp().into();
    });
    let e_x_sum = 1.0 / f64::from(e_x.iter().fold(TenBitExpFP::zero(), |sum, val| sum + *val));
    e_x.iter_mut().for_each(|e| *e *= e_x_sum.into());
    return e_x;
}

pub fn nn_client(
    server_addr: &str,
    architecture: &NeuralArchitecture<TenBitAS, TenBitExpFP>,
    input: Input<TenBitExpFP>,
) -> Input<TenBitExpFP> {
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let client_state = {
       let stream = TcpStream::connect(server_addr).unwrap();
       let read_stream = BufReader::new(stream.try_clone().unwrap());
       let write_stream = stream;
       NNProtocol::offline_client_protocol(read_stream, write_stream, &architecture, &mut rng)
            .unwrap()
    };

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
}

pub fn nn_server(nn: &NeuralNetwork<TenBitAS, TenBitExpFP>, listener: &TcpListener) {
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let server_offline_state = {
        let stream = listener
            .incoming()
            .next()
            .unwrap()
            .expect("server connection failed!");
        let write_stream = stream.try_clone().unwrap();
        let read_stream = BufReader::new(stream);
        NNProtocol::offline_server_protocol(read_stream, write_stream, &nn, &mut rng).unwrap()
    };

    let _ = {
        let stream = listener
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

pub fn run(
    network: NeuralNetwork<TenBitAS, TenBitExpFP>,
    architecture: NeuralArchitecture<TenBitAS, TenBitExpFP>,
    image: Array4<f64>,
    class: i64,
) {
    let server_addr = "127.0.0.1:8001";
    let server_listener = TcpListener::bind(server_addr).unwrap();

    let mut client_output = Output::zeros((1, 10, 0, 0));
    crossbeam::thread::scope(|s| {
        let server_output = s.spawn(|_| nn_server(&network, &server_listener));
        client_output = s.spawn(|_| nn_client(&server_addr, &architecture, (image.clone()).into())).join().unwrap();
        server_output.join().unwrap();
    })
    .unwrap();
    let sm = softmax(&client_output);
    let max = sm.iter().map(|e| f64::from(*e)).fold(0. / 0., f64::max);
    let index = sm.iter().position(|e| f64::from(*e) == max).unwrap() as i64;
    println!("Correct class is {}, inference result is {}", class, index);
}
