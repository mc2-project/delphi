use crate::*;
use algebra::{fields::PrimeField, FpParameters, UniformRandom};
use crypto_primitives::gc::fancy_garbling;
use io_utils::{CountingIO, IMuxSync};
use neural_network::{
    layers::{Layer, NonLinearLayer},
    NeuralNetwork,
};
use ocelot::ot::{AlszSender as OTSender, Sender};
use protocols::{
    gc::ServerGcMsgSend,
    linear_layer::{LinearProtocol, OfflineServerKeyRcv},
};
use protocols_sys::*;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use rayon::prelude::*;
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

fn cg_helper<R: RngCore + CryptoRng>(
    layers: &[usize],
    nn: &NeuralNetwork<TenBitAS, TenBitExpFP>,
    sfhe: ServerFHE,
    reader: &mut IMuxSync<CountingIO<BufReader<TcpStream>>>,
    writer: &mut IMuxSync<CountingIO<BufWriter<TcpStream>>>,
    rng: &mut R,
) {
    let mut linear_state = BTreeMap::new();
    for i in layers.iter() {
        match &nn.layers[*i] {
            Layer::NLL(NonLinearLayer::ReLU(..)) => {}
            Layer::NLL(NonLinearLayer::PolyApprox { .. }) => {}
            Layer::LL(layer) => {
                let randomizer = match &layer {
                    LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                        let mut cg_handler = match &layer {
                            LinearLayer::Conv2d { .. } => SealServerCG::Conv2D(
                                server_cg::Conv2D::new(&sfhe, layer, &layer.kernel_to_repr()),
                            ),
                            LinearLayer::FullyConnected { .. } => {
                                SealServerCG::FullyConnected(server_cg::FullyConnected::new(
                                    &sfhe,
                                    layer,
                                    &layer.kernel_to_repr(),
                                ))
                            }
                            _ => unreachable!(),
                        };
                        LinearProtocol::<TenBitExpParams>::offline_server_protocol(
                            reader,
                            writer,
                            layer.input_dimensions(),
                            layer.output_dimensions(),
                            &mut cg_handler,
                            rng,
                        )
                        .unwrap()
                    }
                    // AvgPool and Identity don't require an offline phase
                    LinearLayer::AvgPool { dims, .. } => Output::zeros(dims.output_dimensions()),
                    LinearLayer::Identity { dims } => Output::zeros(dims.output_dimensions()),
                };
                linear_state.insert(i, randomizer);
            }
        }
    }
}

pub fn cg(server_addr: &str, nn: NeuralNetwork<TenBitAS, TenBitExpFP>) {
    let (mut r1, mut w1) = server_connect(server_addr);
    let (mut r2, mut w2) = server_connect(server_addr);

    let key_time = timer_start!(|| "Keygen");
    let keys: OfflineServerKeyRcv = protocols::bytes::deserialize(&mut r1).unwrap();
    let mut key_share = KeyShare::new();
    let sfhe = key_share.receive(keys.msg());
    timer_end!(key_time);

    let key_time = timer_start!(|| "Keygen");
    let keys: OfflineServerKeyRcv = protocols::bytes::deserialize(&mut r1).unwrap();
    let mut key_share = KeyShare::new();
    let sfhe_2 = key_share.receive(keys.msg());
    timer_end!(key_time);

    r1.reset();

    let (t1_layers, t2_layers) = match nn.layers.len() {
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
        let nn_1 = &nn;
        let nn_2 = &nn;
        s.spawn(move |_| {
            let mut rng = &mut ChaChaRng::from_seed(RANDOMNESS);
            cg_helper(&t1_layers, nn_1, sfhe, r1, w1, &mut rng);
        });

        s.spawn(move |_| {
            let mut rng = &mut ChaChaRng::from_seed(RANDOMNESS);
            cg_helper(&t2_layers, nn_2, sfhe_2, r2, w2, &mut rng);
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
    let (mut reader, mut writer) = server_connect(server_addr);

    let key_time = timer_start!(|| "Keygen");
    let keys: OfflineServerKeyRcv = protocols::bytes::deserialize(&mut reader).unwrap();
    let mut key_share = KeyShare::new();
    let _ = Some(key_share.receive(keys.msg()));
    timer_end!(key_time);
    reader.reset();

    let start_time = timer_start!(|| "ReLU offline protocol");

    let mut gc_s = Vec::with_capacity(number_of_relus);
    let mut encoders = Vec::with_capacity(number_of_relus);
    let p = (<<F as PrimeField>::Params>::MODULUS.0).into();

    let c = protocols::gc::make_relu::<TenBitExpParams>();
    let garble_time = timer_start!(|| "Garbling");
    (0..number_of_relus)
        .into_par_iter()
        .map(|_| {
            let mut c = c.clone();
            let (en, gc) = fancy_garbling::garble(&mut c).unwrap();
            (en, gc)
        })
        .unzip_into_vecs(&mut encoders, &mut gc_s);
    timer_end!(garble_time);

    let encode_time = timer_start!(|| "Encoding inputs");
    let num_garbler_inputs = c.num_garbler_inputs();
    let num_evaluator_inputs = c.num_evaluator_inputs();

    let zero_inputs = vec![0u16; num_evaluator_inputs];
    let one_inputs = vec![1u16; num_evaluator_inputs];
    let mut labels = Vec::with_capacity(number_of_relus * num_evaluator_inputs);
    let mut randomizer_labels = Vec::with_capacity(number_of_relus);
    let mut output_randomizers = Vec::with_capacity(number_of_relus);
    for enc in encoders.iter() {
        let r = F::uniform(rng);
        output_randomizers.push(r);
        let r_bits: u64 = ((-r).into_repr()).into();
        let r_bits =
            fancy_garbling::util::u128_to_bits(r_bits.into(), crypto_primitives::gc::num_bits(p));
        for w in ((num_garbler_inputs / 2)..num_garbler_inputs)
            .zip(r_bits)
            .map(|(i, r_i)| enc.encode_garbler_input(r_i, i))
        {
            randomizer_labels.push(w);
        }

        let all_zeros = enc.encode_evaluator_inputs(&zero_inputs);
        let all_ones = enc.encode_evaluator_inputs(&one_inputs);
        all_zeros
            .into_iter()
            .zip(all_ones)
            .for_each(|(label_0, label_1)| labels.push((label_0.as_block(), label_1.as_block())));
    }
    timer_end!(encode_time);

    let send_gc_time = timer_start!(|| "Sending GCs");
    let randomizer_label_per_relu = if number_of_relus == 0 {
        8192
    } else {
        randomizer_labels.len() / number_of_relus
    };
    for msg_contents in gc_s
        .chunks(8192)
        .zip(randomizer_labels.chunks(randomizer_label_per_relu * 8192))
    {
        let sent_message = ServerGcMsgSend::new(&msg_contents);
        protocols::bytes::serialize(&mut writer, &sent_message).unwrap();
        writer.flush().unwrap();
    }
    timer_end!(send_gc_time);

    add_to_trace!(|| "GC Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
    reader.reset();
    writer.reset();

    if number_of_relus != 0 {
        let r = reader.get_mut_ref().remove(0);
        let w = writer.get_mut_ref().remove(0);

        let ot_time = timer_start!(|| "OTs");
        let mut channel = Channel::new(r, w);
        let mut ot = OTSender::init(&mut channel, rng).unwrap();
        ot.send(&mut channel, labels.as_slice(), rng).unwrap();
        timer_end!(ot_time);
    }

    timer_end!(start_time);
    add_to_trace!(|| "OT Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
}
