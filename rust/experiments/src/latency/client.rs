use crate::*;
use bench_utils::{timer_end, timer_start};
use algebra::{
    fields::PrimeField,
    fixed_point::{FixedPoint, FixedPointParameters},
    fp_64::Fp64Parameters,
    BigInteger64, FpParameters, UniformRandom,
};
use num_traits::identities::Zero;
use crypto_primitives::{
    beavers_mul::{FPBeaversMul, Triple},
    gc::{
        fancy_garbling,
        fancy_garbling::{
            circuit::{Circuit, CircuitBuilder},
            Encoder, GarbledCircuit, Wire,
        },
    },
    Share,
};
use neural_network::{
    layers::{Layer, LayerInfo, LinearLayerInfo, NonLinearLayer, NonLinearLayerInfo},
    NeuralArchitecture, NeuralNetwork,
    tensors::*,
};
use protocols::{
    gc::{ReluProtocol, ClientGcMsgRcv},
    linear_layer::{LinearProtocol, OfflineClientKeySend},
    neural_network::NNProtocol,
};
use std::{
    convert::TryFrom,
    collections::BTreeMap,
    io::{BufReader, BufWriter, Write},
    net::TcpStream,
    sync::{Arc, Mutex},
};
use ocelot::ot::{AlszReceiver as OTReceiver, AlszSender as OTSender, Receiver, Sender};
use protocols_sys::{ClientFHE, ServerFHE, key_share::KeyShare, client_keygen};
use rayon::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use scuttlebutt::Channel;

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4,
    0x76, 0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a,
    0x52, 0xd2,
];

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


fn cg_helper<R: RngCore + CryptoRng>(
    layers: &[usize],
    architecture: &NeuralArchitecture<TenBitAS, TenBitExpFP>,
    mut cfhe: Option<ClientFHE>,
    mut reader: BufReader<TcpStream>,
    mut writer: BufWriter<TcpStream>,
    rng: &mut R,
) {
    let mut in_shares = BTreeMap::new();
    let mut out_shares = BTreeMap::new();
    for i in layers.iter() {
        match &architecture.layers[*i] {
            LayerInfo::NLL(dims, NonLinearLayerInfo::ReLU) => panic!(),
            LayerInfo::NLL(dims, NonLinearLayerInfo::PolyApprox { .. }) => panic!(),
            LayerInfo::LL(dims, linear_layer_info) => {
                let (in_share, mut out_share) = match &linear_layer_info {
                    LinearLayerInfo::Conv2d { .. } | LinearLayerInfo::FullyConnected => {
                        LinearProtocol::<TenBitExpParams>::offline_client_protocol(
                            &mut reader,
                            &mut writer,
                            dims.input_dimensions(),
                            dims.output_dimensions(),
                            &linear_layer_info,
                            rng,
                            &mut cfhe,
                        ).unwrap()
                    },
                    _ => {
                        // AvgPool and Identity don't require an offline communication
                        if out_shares.keys().any(|k| *k == &(i - 1)) {
                            // If the layer comes after a linear layer, apply the function to
                            // the last layer's output share
                            let prev_output_share = out_shares.get(&(i - 1)).unwrap();
                            let mut output_share = Output::zeros(dims.output_dimensions());
                            linear_layer_info
                                .evaluate_naive(prev_output_share, &mut output_share);
                            (Input::zeros(dims.input_dimensions()), output_share)
                        } else {
                            // Otherwise, just return randomizers of 0
                            (
                                Input::zeros(dims.input_dimensions()),
                                Output::zeros(dims.output_dimensions()),
                            )
                        }
                    },
                };

                // We reduce here becase the input to future layers requires
                // shares to already be reduced correctly; for example,
                // `online_server_protocol` reduces at the end of each layer.
                for share in &mut out_share {
                    share.inner.signed_reduce_in_place();
                }
                // r
                in_shares.insert(i, in_share);
                // -(Lr + s)
                out_shares.insert(i, out_share);
            },
        }
    }
}


pub fn cg<R: RngCore + CryptoRng>(
    server_addr: &str,
    architecture: NeuralArchitecture<TenBitAS, TenBitExpFP>,
    rng: &mut R,
) {
    let stream = TcpStream::connect(server_addr).expect("Client connection failed!");
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut writer = BufWriter::new(stream);

    let stream = TcpStream::connect(server_addr).expect("Client connection failed!");
    let mut reader2 = BufReader::new(stream.try_clone().unwrap());
    let mut writer2 = BufWriter::new(stream);

    let key_time = timer_start!(|| "Keygen");
    let mut key_share = KeyShare::new();
    let (cfhe, keys_vec) = key_share.generate();
    let mut cfhe = Some(cfhe);

    let sent_message = OfflineClientKeySend::new(&keys_vec);
    protocols::bytes::serialize(&mut writer, &sent_message).unwrap();
    timer_end!(key_time);

    let key_time = timer_start!(|| "Keygen");
    let mut key_share = KeyShare::new();
    let (cfhe_2, keys_vec) = key_share.generate();
    let mut cfhe_2 = Some(cfhe_2);

    let sent_message = OfflineClientKeySend::new(&keys_vec);
    protocols::bytes::serialize(&mut writer, &sent_message).unwrap();
    timer_end!(key_time);

    // TODO: Wrap in Mutex
//    let mut in_shares = Arc::new(Mutex::new(BTreeMap::new()));
//    let mut out_shares = Arc::new(Mutex::new(BTreeMap::new()));

    // TODO: Divide even and odd between two threads

    let (t1_layers, t2_layers) = match architecture.layers.len() {
        9 => (vec![0, 5, 6], vec![2, 3, 8]),
        17 => (vec![0, 4, 5, 12, 14], vec![2, 7, 9, 10, 16]),
        _ => panic!(),
    };
    
    let linear_time = timer_start!(|| "Linear layers offline phase");
    crossbeam::scope(|s| {
        let architecture_1 = &architecture;
        let architecture_2 = &architecture;
        s.spawn(move |_| {
            let mut rng = &mut ChaChaRng::from_seed(RANDOMNESS);
            cg_helper(&t1_layers, architecture_1, cfhe, reader, writer, &mut rng);
        });
        
        s.spawn(move |_| {
            let mut rng = &mut ChaChaRng::from_seed(RANDOMNESS);
            cg_helper(&t2_layers, architecture_2, cfhe_2, reader2, writer2, &mut rng);
        });
    });
    timer_end!(linear_time);
}

pub fn gc<R: RngCore + CryptoRng>(server_addr: &str, number_of_relus: usize, rng: &mut R) {
    let stream = TcpStream::connect(server_addr).expect("Client connection failed!");
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut writer = BufWriter::new(stream);

    // Keygen
    let key_time = timer_start!(|| "Keygen");
    let mut key_share = KeyShare::new();
    let (cfhe, keys_vec) = key_share.generate();
    let mut cfhe = Some(cfhe);

    let sent_message = OfflineClientKeySend::new(&keys_vec);
    protocols::bytes::serialize(&mut writer, &sent_message).unwrap();
    timer_end!(key_time);

    // Generate dummy labels/layer for CDS
    let shares = vec![AdditiveShare::<TenBitExpParams>::zero(); number_of_relus];

    use fancy_garbling::util::*;
    let start_time = timer_start!(|| "ReLU offline protocol");
    let p = u128::from(<<F as PrimeField>::Params>::MODULUS.0);
    let field_size = crypto_primitives::gc::num_bits(p);

    let rcv_gc_time = timer_start!(|| "Receiving GCs");
    let mut gc_s = Vec::with_capacity(number_of_relus);
    let mut r_wires = Vec::with_capacity(number_of_relus);

    let num_chunks = (number_of_relus as f64 / 8192.0).ceil() as usize;
    for i in 0..num_chunks {
        let in_msg: ClientGcMsgRcv = protocols::bytes::deserialize(&mut reader).unwrap();
        let (gc_chunks, r_wire_chunks) = in_msg.msg();
        if i < (num_chunks - 1) {
            assert_eq!(gc_chunks.len(), 8192);
        }
        gc_s.extend(gc_chunks);
        r_wires.extend(r_wire_chunks);
    }
    timer_end!(rcv_gc_time);

    assert_eq!(gc_s.len(), number_of_relus);
    let bs = shares
        .iter()
        .flat_map(|s| u128_to_bits(protocols::gc::u128_from_share(*s), field_size))
        .map(|b| b == 1)
        .collect::<Vec<_>>();

    let labels = if number_of_relus != 0 {
        let ot_time = timer_start!(|| "OTs");
        let mut channel = Channel::new(&mut reader, &mut writer);
        let mut ot = OTReceiver::init(&mut channel, rng).expect("should work");
        let labels = ot
            .receive(&mut channel, bs.as_slice(), rng)
            .expect("should work");
        let labels = labels
            .into_iter()
            .map(|l| Wire::from_block(l, 2))
            .collect::<Vec<_>>();
        timer_end!(ot_time);
        labels
    } else {
        Vec::new()
    };

    timer_end!(start_time);
}
