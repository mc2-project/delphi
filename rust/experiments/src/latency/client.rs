use crate::*;
use algebra::{fields::PrimeField, FpParameters};
use crypto_primitives::gc::{fancy_garbling, fancy_garbling::Wire};
use io_utils::{counting::CountingIO, imux::IMuxSync};
use neural_network::{
    layers::{LayerInfo, LinearLayerInfo, NonLinearLayerInfo},
    NeuralArchitecture,
};
use num_traits::identities::Zero;
use ocelot::ot::{AlszReceiver as OTReceiver, Receiver};
use protocols::{
    gc::ClientGcMsgRcv,
    linear_layer::{LinearProtocol, OfflineClientKeySend},
};
use protocols_sys::*;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use scuttlebutt::Channel;
use std::{
    collections::BTreeMap,
    io::{BufReader, BufWriter},
    net::TcpStream,
};

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
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

    crate::nn_client(server_addr, &architecture, input, rng);
}

fn cg_helper<R: RngCore + CryptoRng>(
    layers: &[usize],
    architecture: &NeuralArchitecture<TenBitAS, TenBitExpFP>,
    cfhe: ClientFHE,
    reader: &mut IMuxSync<CountingIO<BufReader<TcpStream>>>,
    writer: &mut IMuxSync<CountingIO<BufWriter<TcpStream>>>,
    rng: &mut R,
) {
    let mut in_shares = BTreeMap::new();
    let mut out_shares = BTreeMap::new();
    for i in layers.iter() {
        match &architecture.layers[*i] {
            LayerInfo::NLL(_dims, NonLinearLayerInfo::ReLU) => panic!(),
            LayerInfo::NLL(_dims, NonLinearLayerInfo::PolyApprox { .. }) => panic!(),
            LayerInfo::LL(dims, linear_layer_info) => {
                let input_dims = dims.input_dimensions();
                let output_dims = dims.output_dimensions();
                let (in_share, mut out_share) = match &linear_layer_info {
                    LinearLayerInfo::Conv2d { .. } | LinearLayerInfo::FullyConnected => {
                        let mut cg_handler = match &linear_layer_info {
                            LinearLayerInfo::Conv2d { .. } => {
                                SealClientCG::Conv2D(client_cg::Conv2D::new(
                                    &cfhe,
                                    linear_layer_info,
                                    input_dims,
                                    output_dims,
                                ))
                            }
                            LinearLayerInfo::FullyConnected => {
                                SealClientCG::FullyConnected(client_cg::FullyConnected::new(
                                    &cfhe,
                                    linear_layer_info,
                                    input_dims,
                                    output_dims,
                                ))
                            }
                            _ => unreachable!(),
                        };
                        LinearProtocol::offline_client_protocol(
                            reader,
                            writer,
                            input_dims,
                            output_dims,
                            &mut cg_handler,
                            rng,
                        )
                        .unwrap()
                    }
                    _ => {
                        // AvgPool and Identity don't require an offline communication
                        if out_shares.keys().any(|k| *k == &(i - 1)) {
                            // If the layer comes after a linear layer, apply the function to
                            // the last layer's output share
                            let prev_output_share = out_shares.get(&(i - 1)).unwrap();
                            let mut output_share = Output::zeros(dims.output_dimensions());
                            linear_layer_info.evaluate_naive(prev_output_share, &mut output_share);
                            (Input::zeros(dims.input_dimensions()), output_share)
                        } else {
                            // Otherwise, just return randomizers of 0
                            (
                                Input::zeros(dims.input_dimensions()),
                                Output::zeros(dims.output_dimensions()),
                            )
                        }
                    }
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
            }
        }
    }
}

pub fn cg(server_addr: &str, architecture: NeuralArchitecture<TenBitAS, TenBitExpFP>) {
    let (mut r1, mut w1) = client_connect(server_addr);
    // Give server time to start async listener
    std::thread::sleep_ms(1000);
    let (mut r2, mut w2) = client_connect(server_addr);

    let key_time = timer_start!(|| "Keygen");
    let mut key_share = KeyShare::new();
    let (cfhe, keys_vec) = key_share.generate();

    let sent_message = OfflineClientKeySend::new(&keys_vec);
    protocols::bytes::serialize(&mut w1, &sent_message).unwrap();
    timer_end!(key_time);

    let key_time = timer_start!(|| "Keygen");
    let mut key_share = KeyShare::new();
    let (cfhe_2, keys_vec) = key_share.generate();

    let sent_message = OfflineClientKeySend::new(&keys_vec);
    protocols::bytes::serialize(&mut w1, &sent_message).unwrap();
    timer_end!(key_time);

    w1.reset();

    let (t1_layers, t2_layers) = match architecture.layers.len() {
        9 => (vec![0, 5, 6], vec![2, 3, 8]),
        17 => (vec![0, 4, 5, 12, 14], vec![2, 7, 9, 10, 16]),
        _ => panic!(),
    };

    let linear_time = timer_start!(|| "Linear layers offline phase");
    crossbeam::scope(|s| {
        let r1 = &mut r1;
        let r2 = &mut r2;
        let w1 = &mut w1;
        let w2 = &mut w2;
        let architecture_1 = &architecture;
        let architecture_2 = &architecture;
        s.spawn(move |_| {
            let mut rng = &mut ChaChaRng::from_seed(RANDOMNESS);
            cg_helper(&t1_layers, architecture_1, cfhe, r1, w1, &mut rng);
        });

        s.spawn(move |_| {
            let mut rng = &mut ChaChaRng::from_seed(RANDOMNESS);
            cg_helper(&t2_layers, architecture_2, cfhe_2, r2, w2, &mut rng);
        });
    })
    .unwrap();
    timer_end!(linear_time);
    add_to_trace!(|| "Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        r1.count() + r2.count(),
        w1.count() + w2.count()
    ));
}

pub fn gc<R: RngCore + CryptoRng>(server_addr: &str, number_of_relus: usize, rng: &mut R) {
    let (mut reader, mut writer) = client_connect(server_addr);

    // Keygen
    let key_time = timer_start!(|| "Keygen");
    let mut key_share = KeyShare::new();
    let (_, keys_vec) = key_share.generate();

    let sent_message = OfflineClientKeySend::new(&keys_vec);
    protocols::bytes::serialize(&mut writer, &sent_message).unwrap();
    timer_end!(key_time);

    writer.reset();

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

    add_to_trace!(|| "GC Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
    reader.reset();
    writer.reset();

    assert_eq!(gc_s.len(), number_of_relus);
    let bs = shares
        .iter()
        .flat_map(|s| u128_to_bits(protocols::gc::u128_from_share(*s), field_size))
        .map(|b| b == 1)
        .collect::<Vec<_>>();

    let _ = if number_of_relus != 0 {
        let r = reader.get_mut_ref().remove(0);
        let w = writer.get_mut_ref().remove(0);

        let ot_time = timer_start!(|| "OTs");
        let mut channel = Channel::new(r, w);
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

    add_to_trace!(|| "OT Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
}
