use crate::AdditiveShare;
use algebra::{
    fields::near_mersenne_64::F,
    fixed_point::{FixedPoint, FixedPointParameters},
};
use crypto_primitives::{additive_share::Share, beavers_mul::FPBeaversMul};
use io_utils::IMuxSync;
use protocols_sys::{key_share::KeyShare, ClientFHE, ServerFHE};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use std::net::{TcpListener, TcpStream};

struct TenBitExpParams {}
impl FixedPointParameters for TenBitExpParams {
    type Field = F;
    const MANTISSA_CAPACITY: u8 = 3;
    const EXPONENT_CAPACITY: u8 = 7;
}

type TenBitExpFP = FixedPoint<TenBitExpParams>;
type TenBitBM = FPBeaversMul<TenBitExpParams>;
type TenBitAS = AdditiveShare<TenBitExpParams>;

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

fn generate_random_number<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
    let is_neg: bool = rng.gen();
    let mul = if is_neg { -10.0 } else { 10.0 };
    let mut float: f64 = rng.gen();
    float += 1.0;
    let f = TenBitExpFP::truncate_float(float * mul);
    let n = TenBitExpFP::from(f);
    (f, n)
}

fn generate_random_weight<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
    let is_neg: bool = rng.gen();
    let mul = if is_neg { -10.0 } else { 10.0 };
    let float: f64 = rng.gen_range(-0.9, 0.9);
    let f = TenBitExpFP::truncate_float(float * mul);
    let n = TenBitExpFP::from(f);
    (f, n)
}

mod beavers_mul {
    use super::*;
    use crate::beavers_mul::BeaversMulProtocol;
    use crypto_primitives::Share;

    #[test]
    fn test_beavers_mul() {
        let num_triples = 100;
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        let mut plain_x_s = Vec::with_capacity(num_triples);
        let mut plain_y_s = Vec::with_capacity(num_triples);
        let mut plain_results = Vec::with_capacity(num_triples);

        // Shares for party 1
        let mut x_s_1 = Vec::with_capacity(num_triples);
        let mut y_s_1 = Vec::with_capacity(num_triples);

        // Shares for party 2
        let mut x_s_2 = Vec::with_capacity(num_triples);
        let mut y_s_2 = Vec::with_capacity(num_triples);

        // Give shares to each party
        for _ in 0..num_triples {
            let (f1, n1) = (2.0, TenBitExpFP::from(2.0));
            let (f2, n2) = (5.0, TenBitExpFP::from(5.0));
            plain_x_s.push(n1);
            plain_y_s.push(n2);
            let f3 = f1 * f2;
            let n3 = TenBitExpFP::from(f3);
            plain_results.push(n3);

            let (s11, s12) = n1.share(&mut rng);
            let (s21, s22) = n2.share(&mut rng);
            x_s_1.push(s11);
            x_s_2.push(s12);

            y_s_1.push(s21);
            y_s_2.push(s22);
        }

        // Keygen
        let mut key_share = KeyShare::new();
        let (cfhe, keys_vec) = key_share.generate();
        let sfhe = key_share.receive(keys_vec);

        // Party 1 acts as the server, Party 2 as the client
        let addr = "127.0.0.1:8005";
        let party_1_listener = TcpListener::bind(&addr).unwrap();

        let (triples_1, triples_2) = crossbeam::thread::scope(|s| {
            let triples_1 = s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                for stream in party_1_listener.incoming() {
                    match stream {
                        Ok(read_stream) => {
                            return BeaversMulProtocol::offline_server_protocol::<TenBitBM, _, _, _>(
                                &mut IMuxSync::new(vec![read_stream.try_clone().unwrap()]),
                                &mut IMuxSync::new(vec![read_stream]),
                                &sfhe,
                                num_triples,
                                &mut rng,
                            )
                        },
                        Err(_) => panic!("Connection failed"),
                    }
                }
                unreachable!("we should never exit server's loop")
            });
            let triples_2 = s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                let party_2_stream = TcpStream::connect(&addr).unwrap();
                BeaversMulProtocol::offline_client_protocol::<TenBitBM, _, _, _>(
                    &mut IMuxSync::new(vec![party_2_stream.try_clone().unwrap()]),
                    &mut IMuxSync::new(vec![party_2_stream]),
                    &cfhe,
                    num_triples,
                    &mut rng,
                )
            });
            (
                triples_1.join().unwrap().unwrap(),
                triples_2.join().unwrap().unwrap(),
            )
        })
        .unwrap();
        let (p1, p2) = crossbeam::thread::scope(|s| {
            let p1 = s.spawn(|_| {
                for stream in party_1_listener.incoming() {
                    match stream {
                        Ok(read_stream) => {
                            return BeaversMulProtocol::online_client_protocol::<TenBitBM, _, _>(
                                1, // party index
                                &mut IMuxSync::new(vec![read_stream.try_clone().unwrap()]),
                                &mut IMuxSync::new(vec![read_stream]),
                                &x_s_1,
                                &y_s_1,
                                &triples_1,
                            );
                        },
                        Err(_) => panic!("Connection failed"),
                    }
                }
                unreachable!("we should never exit server's loop")
            });
            let p2 = s.spawn(|_| {
                let party_2_stream = TcpStream::connect(&addr).unwrap();
                BeaversMulProtocol::online_server_protocol::<TenBitBM, _, _>(
                    2, // party index
                    &mut IMuxSync::new(vec![party_2_stream.try_clone().unwrap()]),
                    &mut IMuxSync::new(vec![party_2_stream]),
                    &x_s_2,
                    &y_s_2,
                    &triples_2,
                )
            });
            (p1.join().unwrap().unwrap(), p2.join().unwrap().unwrap())
        })
        .unwrap();
        for (i, ((mut s1, mut s2), n3)) in p1.into_iter().zip(p2).zip(plain_results).enumerate() {
            s1.inner.signed_reduce_in_place();
            s2.inner.signed_reduce_in_place();
            let n4 = s1.combine(&s2);
            assert_eq!(n4, n3, "iteration {} failed", i);
        }
    }
}

mod gc {
    use super::*;
    use crate::gc::*;

    #[test]
    fn test_gc_relu() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let mut plain_x_s = Vec::with_capacity(1001);
        let mut plain_results = Vec::with_capacity(1001);

        // Shares for server
        let mut server_x_s = Vec::with_capacity(1001);
        let mut randomizer = Vec::with_capacity(1001);

        // Shares for client
        let mut client_x_s = Vec::with_capacity(1001);
        let mut client_results = Vec::with_capacity(1001);

        for _ in 0..1000 {
            let (f1, n1) = generate_random_number(&mut rng);
            plain_x_s.push(n1);
            let f2 = if f1 < 0.0 {
                0.0
            } else if f1 > 6.0 {
                6.0
            } else {
                f1
            };
            let n2 = TenBitExpFP::from(f2);
            plain_results.push(n2);

            let (s11, s12) = n1.share(&mut rng);
            let (_, s22) = n2.share(&mut rng);
            server_x_s.push(s11);
            client_x_s.push(s12);

            randomizer.push(-s22.inner.inner);
            client_results.push(s22);
        }

        let server_addr = "127.0.0.1:8003";
        let client_addr = "127.0.0.1:8004";
        let num_relus = 1000;
        let server_listener = TcpListener::bind(server_addr).unwrap();

        let (server_offline, client_offline) = crossbeam::thread::scope(|s| {
            let server_offline_result = s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);

                for stream in server_listener.incoming() {
                    let stream = stream.expect("server connection failed!");
                    let mut read_stream = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                    let mut write_stream = IMuxSync::new(vec![stream]);
                    return ReluProtocol::<TenBitExpParams>::offline_server_protocol(
                        &mut read_stream,
                        &mut write_stream,
                        num_relus,
                        &mut rng,
                    );
                }
                unreachable!("we should never exit server's loop")
            });

            let client_offline_result = s.spawn(|_| {
                let mut rng = ChaChaRng::from_seed(RANDOMNESS);

                // client's connection to server.
                let stream = TcpStream::connect(server_addr).expect("connecting to server failed");
                let mut read_stream = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                let mut write_stream = IMuxSync::new(vec![stream]);

                return ReluProtocol::offline_client_protocol(
                    &mut read_stream,
                    &mut write_stream,
                    num_relus,
                    &client_x_s,
                    &mut rng,
                );
            });
            (
                server_offline_result.join().unwrap().unwrap(),
                client_offline_result.join().unwrap().unwrap(),
            )
        })
        .unwrap();
        let client_listener = TcpListener::bind(client_addr).unwrap();

        let client_online = crossbeam::thread::scope(|s| {
            // Start thread for client.
            let result = s.spawn(|_| {
                let gc_s = &client_offline.gc_s;
                let server_labels = &client_offline.server_randomizer_labels;
                let client_labels = &client_offline.client_input_labels;
                for stream in client_listener.incoming() {
                    let mut read_stream =
                        IMuxSync::new(vec![stream.expect("client connection failed!")]);
                    return ReluProtocol::online_client_protocol(
                        &mut read_stream,
                        num_relus,
                        &server_labels,
                        &client_labels,
                        &gc_s,
                        &randomizer,
                    );
                }
                unreachable!("we should never reach here")
            });

            // Start thread for the server to make a connection.
            let _ = s
                .spawn(|_| {
                    let mut write_stream =
                        IMuxSync::new(vec![TcpStream::connect(client_addr).unwrap()]);

                    ReluProtocol::online_server_protocol(
                        &mut write_stream,
                        &server_x_s,
                        &server_offline.encoders,
                    )
                })
                .join()
                .unwrap();

            result.join().unwrap().unwrap()
        })
        .unwrap();
        for i in 0..1000 {
            let server_randomizer = server_offline.output_randomizers[i];
            let server_share =
                TenBitExpFP::randomize_local_share(&client_online[i], &server_randomizer);
            let client_share = client_results[i];
            let result = plain_results[i];
            assert_eq!(server_share.combine(&client_share), result);
        }
    }
}

mod linear {
    use super::*;
    use crate::linear_layer::*;
    use ndarray::s;
    use neural_network::{layers::*, tensors::*, Evaluate};
    use std::io::{BufReader, BufWriter};

    #[test]
    fn test_convolution() {
        use neural_network::layers::convolution::*;

        const RANDOMNESS: [u8; 32] = [
            0x14, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda,
            0xf4, 0x76, 0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77,
            0xd3, 0x4a, 0x52, 0xd2,
        ];

        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        // Set the parameters for the convolution.
        let input_dims = (1, 64, 8, 8);
        let kernel_dims = (64, 64, 3, 3);
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

        let layer_params =
            Conv2dParams::<TenBitAS, _>::new(padding, stride, kernel.clone(), bias.clone());
        let output_dims = layer_params.calculate_output_size(input_dims);
        let pt_layer_params =
            Conv2dParams::<TenBitExpFP, _>::new(padding, stride, kernel.clone(), bias.clone());
        let layer_dims = LayerDims {
            input_dims,
            output_dims,
        };
        let layer = Layer::LL(LinearLayer::Conv2d {
            dims:   layer_dims,
            params: layer_params,
        });
        let layer_info = (&layer).into();
        let layer = match layer {
            Layer::LL(l) => l,
            Layer::NLL(_) => unreachable!(),
        };
        let pt_layer = LinearLayer::Conv2d {
            dims:   layer_dims,
            params: pt_layer_params,
        };
        // Done setting up parameters for the convolution

        // Sample a random input.
        let mut input = Input::zeros(input_dims);
        input
            .iter_mut()
            .for_each(|in_i| *in_i = generate_random_number(&mut rng).1);

        // Evaluate convolution layer on plaintext, so that we can check results later.
        let output = pt_layer.evaluate(&input);

        let server_addr = "127.0.0.1:8001";
        let server_listener = TcpListener::bind(server_addr).unwrap();

        let layer_input_dims = layer.input_dimensions();
        let layer_output_dims = layer.output_dimensions();
        let layer = std::sync::Arc::new(std::sync::Mutex::new(layer));
        let ((layer_randomizer, client_next_layer_share), server_offline) =
            crossbeam::thread::scope(|s| {
                let server_offline_result = s.spawn(|_| {
                    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                    let mut sfhe_op: Option<ServerFHE> = None;

                    for stream in server_listener.incoming() {
                        let stream = stream.expect("server connection failed!");
                        // let mut write_stream = read_stream.try_clone().unwrap();
                        return LinearProtocol::offline_server_protocol(
                            &mut IMuxSync::new(vec![BufReader::new(&stream)]),
                            &mut IMuxSync::new(vec![BufWriter::new(&stream)]),
                            &layer.lock().unwrap(), // layer parameters
                            &mut rng,
                            &mut sfhe_op,
                        );
                    }
                    unreachable!("we should never exit server's loop")
                });

                let client_offline_result = s.spawn(|_| {
                    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                    let mut cfhe_op: Option<ClientFHE> = None;

                    // client's connection to server.
                    // TODO: Figure out why BufStream doesn't work here
                    let stream =
                        TcpStream::connect(server_addr).expect("connecting to server failed");
                    let mut read_stream = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                    let mut write_stream = IMuxSync::new(vec![stream]);

                    match &layer_info {
                        LayerInfo::LL(_, info) => LinearProtocol::offline_client_protocol(
                            &mut read_stream,
                            &mut write_stream,
                            layer_input_dims,
                            layer_output_dims,
                            &info,
                            &mut rng,
                            &mut cfhe_op,
                        ),
                        LayerInfo::NLL(..) => unreachable!(),
                    }
                });
                (
                    client_offline_result.join().unwrap().unwrap(),
                    server_offline_result.join().unwrap().unwrap(),
                )
            })
            .unwrap();

        // Share the input for layer `1`, computing
        // server_share_1 = x + r.
        // client_share_1 = -r;
        let (server_current_layer_share, _) = input.share_with_randomness(&layer_randomizer);

        let server_next_layer_share = crossbeam::thread::scope(|s| {
            // Start thread for client.
            let result = s.spawn(|_| {
                let mut write_stream =
                    IMuxSync::new(vec![TcpStream::connect(server_addr).unwrap()]);
                let mut result = Output::zeros(layer_output_dims);
                match &layer_info {
                    LayerInfo::LL(_, info) => LinearProtocol::online_client_protocol(
                        &mut write_stream,
                        &server_current_layer_share,
                        &info,
                        &mut result,
                    ),
                    LayerInfo::NLL(..) => unreachable!(),
                }
            });

            // Start thread for the server to make a connection.
            let server_result = s
                .spawn(move |_| {
                    for stream in server_listener.incoming() {
                        let mut read_stream =
                            IMuxSync::new(vec![stream.expect("server connection failed!")]);
                        let mut output = Output::zeros(output_dims);
                        return LinearProtocol::online_server_protocol(
                            &mut read_stream,       // we only receive here, no messages to client
                            &layer.lock().unwrap(), // layer parameters
                            &server_offline,        // this is our `s` from above.
                            &Input::zeros(layer_input_dims),
                            &mut output, // this is where the result will go.
                        )
                        .map(|_| output);
                    }
                    unreachable!("Server should not exit loop");
                })
                .join()
                .unwrap()
                .unwrap();

            let _ = result.join();
            server_result
        })
        .unwrap();

        let mut result: Input<TenBitExpFP> = Input::zeros(client_next_layer_share.dim());
        result
            .iter_mut()
            .zip(client_next_layer_share.iter())
            .zip(server_next_layer_share.iter())
            .for_each(|((r, s1), s2)| {
                *r = (*s1).combine(s2);
            });

        println!("Result:");
        println!("DIM: {:?}", result.dim());
        let chan_size = result.dim().2 * result.dim().3;
        let row_size = result.dim().2;
        let mut success = true;
        result
            .slice(s![0, .., .., ..])
            .outer_iter()
            .zip(output.slice(s![0, .., .., ..]).outer_iter())
            .enumerate()
            .for_each(|(chan_idx, (res_c, out_c))| {
                println!("Channel {}: ", chan_idx);

                res_c
                    .outer_iter()
                    .zip(out_c.outer_iter())
                    .enumerate()
                    .for_each(|(inp_idx, (inp_r, inp_out))| {
                        println!("    Row {}: ", inp_idx);

                        inp_r
                            .iter()
                            .zip(inp_out.iter())
                            .enumerate()
                            .for_each(|(i, (r, out))| {
                                println!(
                                    "IDX {}:           {}        {}",
                                    i + inp_idx * row_size + chan_idx * chan_size,
                                    r,
                                    out
                                );
                                let delta = f64::from(*r) - f64::from(*out);
                                if delta.abs() > 0.5 {
                                    println!(
                                        "{:?}-th index failed {:?} {:?} {} {}",
                                        i,
                                        r.signed_reduce(),
                                        out.signed_reduce(),
                                        r,
                                        out
                                    );
                                    println!(
                                        "{} + {} = {}",
                                        client_next_layer_share[[0, chan_idx, inp_idx, i]].inner,
                                        server_next_layer_share[[0, chan_idx, inp_idx, i]].inner,
                                        r
                                    );
                                    success = false;
                                }
                            });
                    });
            });
        assert!(success);
    }

    #[test]
    fn test_fully_connected() {
        use neural_network::layers::fully_connected::*;

        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        // Set the parameters for the layer
        let input_dims = (1, 3, 32, 32);
        let kernel_dims = (10, 3, 32, 32);

        // Sample a random kernel.
        let mut kernel = Kernel::zeros(kernel_dims);
        let mut bias = Kernel::zeros((kernel_dims.0, 1, 1, 1));
        kernel
            .iter_mut()
            .for_each(|ker_i| *ker_i = generate_random_weight(&mut rng).1);
        bias.iter_mut()
            .for_each(|bias_i| *bias_i = generate_random_weight(&mut rng).1);

        let layer_params = FullyConnectedParams::<TenBitAS, _>::new(kernel.clone(), bias.clone());
        let pt_layer_params =
            FullyConnectedParams::<TenBitExpFP, _>::new(kernel.clone(), bias.clone());
        let output_dims = layer_params.calculate_output_size(input_dims);
        let layer_dims = LayerDims {
            input_dims,
            output_dims,
        };
        let layer = Layer::LL(LinearLayer::FullyConnected {
            dims:   layer_dims,
            params: layer_params,
        });
        let layer_info = (&layer).into();
        let layer = match layer {
            Layer::LL(l) => l,
            Layer::NLL(_) => unreachable!(),
        };
        let pt_layer = LinearLayer::FullyConnected {
            dims:   layer_dims,
            params: pt_layer_params,
        };

        // Sample a random input.
        let mut input = Input::zeros(input_dims);
        input
            .iter_mut()
            .for_each(|in_i| *in_i = generate_random_weight(&mut rng).1);
        // input.iter_mut().for_each(|in_i|  *in_i = TenBitExpFP::from(1.0));
        // Evaluate convolution layer on plaintext, so that we can check results later.
        let output = pt_layer.evaluate(&input);

        let server_addr = "127.0.0.1:8002";
        let server_listener = TcpListener::bind(server_addr).unwrap();

        let layer_input_dims = layer.input_dimensions();
        let layer_output_dims = layer.output_dimensions();
        let layer = std::sync::Arc::new(std::sync::Mutex::new(layer));
        let ((layer_randomizer, client_next_layer_share), server_offline) =
            crossbeam::thread::scope(|s| {
                let server_offline_result = s.spawn(|_| {
                    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                    let mut sfhe_op: Option<ServerFHE> = None;

                    for stream in server_listener.incoming() {
                        let stream = stream.expect("server connection failed!");
                        // let mut write_stream = read_stream.try_clone().unwrap();
                        return LinearProtocol::offline_server_protocol(
                            &mut IMuxSync::new(vec![BufReader::new(&stream)]),
                            &mut IMuxSync::new(vec![BufWriter::new(&stream)]),
                            &layer.lock().unwrap(), // layer parameters
                            &mut rng,
                            &mut sfhe_op,
                        );
                    }
                    unreachable!("we should never exit server's loop")
                });

                let client_offline_result = s.spawn(|_| {
                    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
                    let mut cfhe_op: Option<ClientFHE> = None;

                    // client's connection to server.
                    // TODO: Figure out why BufStream doesn't work here
                    let stream =
                        TcpStream::connect(server_addr).expect("connecting to server failed");
                    let mut read_stream = IMuxSync::new(vec![stream.try_clone().unwrap()]);
                    let mut write_stream = IMuxSync::new(vec![stream]);

                    match &layer_info {
                        LayerInfo::LL(_, info) => LinearProtocol::offline_client_protocol(
                            &mut read_stream,
                            &mut write_stream,
                            layer_input_dims,
                            layer_output_dims,
                            &info,
                            &mut rng,
                            &mut cfhe_op,
                        ),
                        LayerInfo::NLL(..) => unreachable!(),
                    }
                });
                (
                    client_offline_result.join().unwrap().unwrap(),
                    server_offline_result.join().unwrap().unwrap(),
                )
            })
            .unwrap();

        println!("\nSERVER'S SHARE: ");
        server_offline.iter().for_each(|e| {
            let as_share: FixedPoint<TenBitExpParams> = -(FixedPoint::with_num_muls(*e, 1));
            println!("{} {}", as_share.inner, as_share);
        });
        println!("\n");

        println!("CLIENT'S NEXT LAYER SHARE:");
        client_next_layer_share.iter().for_each(|e| {
            println!("{}, {}", e.inner.inner, e.inner);
        });
        println!("\n");

        println!("CLIENT'S LAYER RANDOMIZER:");
        layer_randomizer.iter().for_each(|e: &F| {
            let as_share: FixedPoint<TenBitExpParams> = FixedPoint::with_num_muls(*e, 0);
            println!("{}, {}", e, as_share);
        });
        println!("\n");

        // Share the input for layer `1`, computing
        // server_share_1 = x + r.
        // client_share_1 = -r;
        let (server_current_layer_share, _) = input.share_with_randomness(&layer_randomizer);

        println!("CLIENT ONLINE INPUT:");
        server_current_layer_share.iter().for_each(|e| {
            println!("{}, {}", e.inner.inner, e.inner);
        });
        println!("\n");

        let server_next_layer_share = crossbeam::thread::scope(|s| {
            // Start thread for client.
            let result = s.spawn(|_| {
                let mut write_stream =
                    IMuxSync::new(vec![TcpStream::connect(server_addr).unwrap()]);
                let mut result = Output::zeros(layer_output_dims);
                match &layer_info {
                    LayerInfo::LL(_, info) => LinearProtocol::online_client_protocol(
                        &mut write_stream,
                        &server_current_layer_share,
                        &info,
                        &mut result,
                    ),
                    LayerInfo::NLL(..) => unreachable!(),
                }
            });

            // Start thread for the server to make a connection.
            let server_result = s
                .spawn(move |_| {
                    for stream in server_listener.incoming() {
                        let mut read_stream =
                            IMuxSync::new(vec![stream.expect("server connection failed!")]);
                        let mut output = Output::zeros(output_dims);
                        return LinearProtocol::online_server_protocol(
                            &mut read_stream,       // we only receive here, no messages to client
                            &layer.lock().unwrap(), // layer parameters
                            &server_offline,        // this is our `s` from above.
                            &Input::zeros(layer_input_dims),
                            &mut output, // this is where the result will go.
                        )
                        .map(|_| output);
                    }
                    unreachable!("Server should not exit loop");
                })
                .join()
                .unwrap()
                .unwrap();

            let _ = result.join();
            server_result
        })
        .unwrap();

        println!("\nSERVER ONLINE OUTPUT:");
        server_next_layer_share.iter().for_each(|e| {
            println!("{}, {}", e.inner.inner, e.inner);
        });
        println!("\n");

        println!("CLIENT UNMASKING:");
        for (i, ((s1, s2), &n3)) in client_next_layer_share
            .iter()
            .zip(server_next_layer_share.iter())
            .zip(output.iter())
            .enumerate()
        {
            let s1 = *s1;
            let s2 = *s2;
            let n4 = s1.combine(&s2);
            println!("{} + {} = {}", s1.inner, s2.inner, n4);
            assert_eq!(n4, n3, "{:?}-th index failed", i);
        }
    }
}
