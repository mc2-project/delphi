use crate::*;
use neural_network::{ndarray::Array4, tensors::Input, NeuralArchitecture};
use protocols::neural_network::NNProtocol;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use std::{
    cmp,
    io::BufReader,
    net::{TcpListener, TcpStream},
    sync::atomic::{AtomicUsize, Ordering},
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
        // client's connection to server.
        // A bit hacky but just in case the server thread doesn't spawn in time
        let mut ctr = 0;
        let stream = loop {
            if let Ok(stream) = TcpStream::connect(server_addr) {
                break stream;
            }
            if ctr > 10 {
                panic!("Server connection refused")
            }
            ctr += 1;
            std::thread::sleep(std::time::Duration::from_millis(100));
        };
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

pub fn nn_server(server_addr: &str, nn: &NeuralNetwork<TenBitAS, TenBitExpFP>) {
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let server_listener = TcpListener::bind(server_addr).unwrap();
    let server_offline_state = {
        let stream = server_listener
            .incoming()
            .next()
            .unwrap()
            .expect("server connection failed!");
        let write_stream = stream.try_clone().unwrap();
        let read_stream = BufReader::new(stream);
        NNProtocol::offline_server_protocol(read_stream, write_stream, &nn, &mut rng).unwrap()
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

pub fn run(
    network: NeuralNetwork<TenBitAS, TenBitExpFP>,
    architecture: NeuralArchitecture<TenBitAS, TenBitExpFP>,
    images: Vec<Array4<f64>>,
    classes: Vec<i64>,
    plaintext: Vec<i64>,
) {
    let base_port = 8001;
    let image_idx = AtomicUsize::new(0);
    let port_idx = AtomicUsize::new(0);
    let correct = AtomicUsize::new(0);
    let correct_pt = AtomicUsize::new(0);
    let cat_failures = AtomicUsize::new(0);
    let non_cat_failures = AtomicUsize::new(0);

    let thread_fn = || {
        let i = image_idx.fetch_add(1, Ordering::SeqCst);
        if i >= images.len() {
            return;
        }
        let port_off = port_idx.fetch_add(1, Ordering::SeqCst);
        let server_addr = format!("127.0.0.1:{}", base_port + port_off);

        let mut client_output = Output::zeros((1, 10, 0, 0));
        crossbeam::thread::scope(|s| {
            let server_output = s.spawn(|_| nn_server(&server_addr, &network));
            client_output = nn_client(&server_addr, &architecture, (images[i].clone()).into());
            server_output.join().unwrap();
        })
        .unwrap();
        let sm = softmax(&client_output);
        let max = sm.iter().map(|e| f64::from(*e)).fold(0. / 0., f64::max);
        let index = sm.iter().position(|e| f64::from(*e) == max).unwrap() as i64;
        // Check the l1 norm of the resulting vector. If it's above 50000 we probably
        // had a catastrophic failure so tally that
        let mut big_fail = false;
        if client_output
            .iter()
            .fold(0.0, |acc, x| acc + f64::from(*x).abs())
            > 5000.0
        {
            big_fail = true;
            cat_failures.fetch_add(1, Ordering::SeqCst);
        }
        if index == classes[i] {
            correct.fetch_add(1, Ordering::SeqCst);
        }
        if index == classes[i] && (plaintext[i] == 1) {
            correct_pt.fetch_add(1, Ordering::SeqCst);
        }
        if (index == classes[i] && plaintext[i] == 0) || (index != classes[i] && plaintext[i] == 1)
        {
            println!(
                "DIFFERED ON IMAGE {} - Correct is {}, and plaintext is {}",
                i, classes[i], plaintext[i]
            );
            if !big_fail {
                non_cat_failures.fetch_add(1, Ordering::SeqCst);
                println!("Protocol result: [");
                for result in &client_output {
                    println!("  {:?}, {}", result, result);
                }
                println!("Softmax:");
                for result in &sm {
                    println!("  {:?}, {}", result, result);
                }
                println!("]");
                println!("Out: {}", index);
            }
        } else {
            println!("IMAGE {} CORRECT!", i);
        }
    };
    // Only spawn as many threads as will fit on the cores
    let num_threads = num_cpus::get() / 2;
    for _ in (0..images.len()).step_by(num_threads) {
        crossbeam::thread::scope(|s| {
            for _ in 0..num_threads {
                s.spawn(|_| thread_fn()).join().unwrap();
            }
        })
        .unwrap();
        port_idx.fetch_sub(num_threads, Ordering::SeqCst);
    }
    let correct = correct.into_inner();
    let correct_pt = correct_pt.into_inner();
    let cat_failures = cat_failures.into_inner();
    let non_cat_failures = non_cat_failures.into_inner();
    println!("Overall Correct: {}", correct);
    println!("Plaintext Correct: {}", correct_pt);
    println!("Catastrophic Failures: {}", cat_failures);
    println!("Non-Catastrophic Failures: {}", non_cat_failures);
}
