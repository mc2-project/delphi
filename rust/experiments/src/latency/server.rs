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
    gc::{ReluProtocol, ClientGcMsgRcv, ServerGcMsgSend},
    linear_layer::{LinearProtocol, OfflineServerKeyRcv},
    neural_network::NNProtocol,
};
use std::{
    convert::TryFrom,
    collections::BTreeMap,
    io::{BufReader, BufWriter, Write},
    net::{TcpStream, TcpListener},
};
use ocelot::ot::{AlszReceiver as OTReceiver, AlszSender as OTSender, Receiver, Sender};
use protocols_sys::{ClientFHE, ServerFHE, key_share::KeyShare, client_keygen};
use rayon::prelude::*;
use scuttlebutt::Channel;

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

pub fn cg<R: RngCore + CryptoRng>(
    server_addr: &str,
    nn: NeuralNetwork<TenBitAS, TenBitExpFP>,
    rng: &mut R,
) {
    let listener = TcpListener::bind(server_addr).unwrap();
    let stream = listener
        .incoming()
        .next()
        .unwrap()
        .expect("server connection failed!");
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut writer = BufWriter::new(stream);

    let key_time = timer_start!(|| "Keygen");
    let keys: OfflineServerKeyRcv = protocols::bytes::deserialize(&mut reader).unwrap();
    let mut key_share = KeyShare::new();
    let mut sfhe = Some(key_share.receive(keys.msg()));
    timer_end!(key_time);

    let mut linear_state = BTreeMap::new();

    let linear_time = timer_start!(|| "Linear layers offline phase");
    for (i, layer) in nn.layers.iter().enumerate() {
        match layer {
            Layer::NLL(NonLinearLayer::ReLU(dims)) => { },
            Layer::NLL(NonLinearLayer::PolyApprox { dims, .. }) => { },
            Layer::LL(layer) => {
                let randomizer = match &layer {
                    LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                        LinearProtocol::offline_server_protocol(
                            &mut reader,
                            &mut writer,
                            &layer,
                            rng,
                            &mut sfhe,
                        ).unwrap()
                    },
                    // AvgPool and Identity don't require an offline phase
                    LinearLayer::AvgPool { dims, .. } => {
                        Output::zeros(dims.output_dimensions())
                    },
                    LinearLayer::Identity { dims } => Output::zeros(dims.output_dimensions()),
                };
                linear_state.insert(i, randomizer);
            },
        }
    }
    timer_end!(linear_time);
}


pub fn gc<R: RngCore + CryptoRng>(server_addr: &str, number_of_relus: usize, rng: &mut R) {
    let listener = TcpListener::bind(server_addr).unwrap();
    let stream = listener
        .incoming()
        .next()
        .unwrap()
        .expect("server connection failed!");
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut writer = BufWriter::new(stream);

    let key_time = timer_start!(|| "Keygen");
    let keys: OfflineServerKeyRcv = protocols::bytes::deserialize(&mut reader).unwrap();
    let mut key_share = KeyShare::new();
    let mut sfhe = Some(key_share.receive(keys.msg()));
    timer_end!(key_time);

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
        let r_bits = fancy_garbling::util::u128_to_bits(
            r_bits.into(),
            crypto_primitives::gc::num_bits(p),
        );
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
            .for_each(|(label_0, label_1)| {
                labels.push((label_0.as_block(), label_1.as_block()))
            });
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

    if number_of_relus != 0 {
        let ot_time = timer_start!(|| "OTs");
        let mut channel = Channel::new(&mut reader, &mut writer);
        let mut ot = OTSender::init(&mut channel, rng).unwrap();
        ot.send(&mut channel, labels.as_slice(), rng).unwrap();
        timer_end!(ot_time);
    }

    timer_end!(start_time);
}
