use crate::AdditiveShare;
use bench_utils::{timer_end, timer_start};
use neural_network::{
    layers::{Layer, LayerInfo, NonLinearLayer, NonLinearLayerInfo},
    NeuralArchitecture, NeuralNetwork,
};
use rand::{CryptoRng, RngCore};
use std::{
    io::{Read, Write},
    marker::PhantomData,
};

use algebra::{
    fixed_point::{FixedPoint, FixedPointParameters},
    fp_64::Fp64Parameters,
    FpParameters, PrimeField,
};

use neural_network::{
    layers::*,
    tensors::{Input, Output},
};

use crypto_primitives::{
    beavers_mul::{FPBeaversMul, Triple},
    gc::fancy_garbling::{Encoder, GarbledCircuit, Wire},
};

use crate::{gc::ReluProtocol, linear_layer::LinearProtocol, quad_approx::QuadApproxProtocol};
use protocols_sys::{ClientFHE, ServerFHE};
use std::collections::BTreeMap;

pub struct NNProtocol<P: FixedPointParameters> {
    _share: PhantomData<P>,
}

pub const CLIENT: usize = 1;
pub const SERVER: usize = 2;

pub struct ServerState<P: FixedPointParameters> {
    pub linear_state:            BTreeMap<usize, Output<P::Field>>,
    pub relu_encoders:           Vec<Encoder>,
    pub relu_output_randomizers: Vec<P::Field>,
    pub approx_state:            Vec<Triple<P::Field>>,
}
// This is a hack since Send + Sync aren't implemented for the raw pointer types
// Not sure if there's a cleaner way to guarantee this
unsafe impl<P: FixedPointParameters> Send for ServerState<P> {}
unsafe impl<P: FixedPointParameters> Sync for ServerState<P> {}

pub struct ClientState<P: FixedPointParameters> {
    pub relu_circuits:                 Vec<GarbledCircuit>,
    pub relu_server_labels:            Vec<Vec<Wire>>,
    pub relu_client_labels:            Vec<Vec<Wire>>,
    pub relu_next_layer_randomizers:   Vec<P::Field>,
    pub approx_state:                  Vec<Triple<P::Field>>,
    /// Randomizers for the input of a linear layer.
    pub linear_randomizer:             BTreeMap<usize, Input<P::Field>>,
    /// Shares of the output of a linear layer
    pub linear_post_application_share: BTreeMap<usize, Output<AdditiveShare<P>>>,
}

pub struct NNProtocolType;
// The final message from the server to the client, contains a share of the
// output.
pub type MsgSend<'a, P> = crate::OutMessage<'a, Output<AdditiveShare<P>>, NNProtocolType>;
pub type MsgRcv<P> = crate::InMessage<Output<AdditiveShare<P>>, NNProtocolType>;

/// ```markdown
///                   Client                     Server
/// --------------------------------------------------------------------------
/// --------------------------------------------------------------------------
/// Offline:
/// 1. Linear:
///                 1. Sample randomizers r
///                 for each layer.
///
///                       ------- Enc(r) ------>
///                                              1. Sample randomness s_1.
///                                              2. Compute Enc(Mr + s_1)
///                       <--- Enc(Mr + s_1) ---
///                 2. Store -(Mr + s1)
///
/// 2. ReLU:
///                                              1. Sample online output randomizers s_2
///                                              2. Garble ReLU circuit with s_2 as input.
///                       <-------- GC ---------
///                 1. OT input:
///                     Mr_i + s_(1, i),
///                     r_{i + 1}
///                       <-------- OT -------->
///
/// 3. Quadratic approx:
///                       <- Beaver's Triples ->
///
/// --------------------------------------------------------------------------
///
/// Online:
///
/// 1. Linear:
///                       -- x_i + r_i + s_{2, i} ->
///
///
///                                               1. Derandomize the input
///                                               1. Compute y_i = M(x_i + r_i) + s_{1, i}
///
/// 2. ReLU:
///                                               2. Compute garbled labels for y_i
///                       <- garbled labels -----
///                 1. Evaluate garbled circuit,
///                 2. Set next layer input to
///                 be output of GC.
///
/// 3. Quad Approx
///                   ---- (multiplication protocol) ----
///                  |                                  |
///                  ▼                                  ▼
///                y_i + a                              a
///
///                       ------ y_i + a + r_i -->
/// ```
impl<P: FixedPointParameters> NNProtocol<P>
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    pub fn offline_server_protocol<R: Read, W: Write, RNG: CryptoRng + RngCore>(
        mut reader: R,
        mut writer: W,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
    ) -> Result<ServerState<P>, bincode::Error> {
        let mut num_relu = 0;
        let mut num_approx = 0;
        let mut linear_state = BTreeMap::new();
        let mut sfhe_op: Option<ServerFHE> = None;

        let start_time = timer_start!(|| "Server offline phase");
        let linear_time = timer_start!(|| "Linear layers offline phase");
        for (i, layer) in neural_network.layers.iter().enumerate() {
            match layer {
                Layer::NLL(NonLinearLayer::ReLU(dims)) => {
                    let (b, c, h, w) = dims.input_dimensions();
                    num_relu += b * c * h * w;
                },
                Layer::NLL(NonLinearLayer::PolyApprox { dims, .. }) => {
                    let (b, c, h, w) = dims.input_dimensions();
                    num_approx += b * c * h * w;
                },
                Layer::LL(layer) => {
                    let randomizer = match &layer {
                        LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                            LinearProtocol::offline_server_protocol(
                                &mut reader,
                                &mut writer,
                                &layer,
                                rng,
                                &mut sfhe_op,
                            )?
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

        let relu_time =
            timer_start!(|| format!("ReLU layers offline phase, with {:?} activations", num_relu));
        let crate::gc::ServerState {
            encoders: relu_encoders,
            output_randomizers: relu_output_randomizers,
        } = ReluProtocol::<P>::offline_server_protocol(&mut reader, &mut writer, num_relu, rng)?;
        timer_end!(relu_time);

        let approx_time = timer_start!(|| format!(
            "Approx layers offline phase, with {:?} activations",
            num_approx
        ));
        let approx_state = QuadApproxProtocol::offline_server_protocol::<FPBeaversMul<P>, _, _, _>(
            &mut reader,
            &mut writer,
            &(sfhe_op.unwrap()),
            num_approx,
            rng,
        )?;
        timer_end!(approx_time);
        timer_end!(start_time);
        Ok(ServerState {
            linear_state,
            relu_encoders,
            relu_output_randomizers,
            approx_state,
        })
    }

    pub fn offline_client_protocol<R: Read, W: Write, RNG: RngCore + CryptoRng>(
        mut reader: R,
        mut writer: W,
        neural_network_architecture: &NeuralArchitecture<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
    ) -> Result<ClientState<P>, bincode::Error> {
        let mut num_relu = 0;
        let mut num_approx = 0;
        let mut in_shares = BTreeMap::new();
        let mut out_shares = BTreeMap::new();
        let mut relu_layers = Vec::new();
        let mut approx_layers = Vec::new();
        let mut cfhe_op: Option<ClientFHE> = None;

        let start_time = timer_start!(|| "Client offline phase");
        let linear_time = timer_start!(|| "Linear layers offline phase");
        for (i, layer) in neural_network_architecture.layers.iter().enumerate() {
            match layer {
                LayerInfo::NLL(dims, NonLinearLayerInfo::ReLU) => {
                    relu_layers.push(i);
                    let (b, c, h, w) = dims.input_dimensions();
                    num_relu += b * c * h * w;
                },
                LayerInfo::NLL(dims, NonLinearLayerInfo::PolyApprox { .. }) => {
                    approx_layers.push(i);
                    let (b, c, h, w) = dims.input_dimensions();
                    num_approx += b * c * h * w;
                },
                LayerInfo::LL(dims, linear_layer_info) => {
                    let (in_share, mut out_share) = match &linear_layer_info {
                        LinearLayerInfo::Conv2d { .. } | LinearLayerInfo::FullyConnected => {
                            LinearProtocol::<P>::offline_client_protocol(
                                &mut reader,
                                &mut writer,
                                dims.input_dimensions(),
                                dims.output_dimensions(),
                                &linear_layer_info,
                                rng,
                                &mut cfhe_op,
                            )?
                        },
                        _ => {
                            // AvgPool and Identity don't require an offline communication
                            if out_shares.keys().any(|k| k == &(i - 1)) {
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
        timer_end!(linear_time);
        // Preprocessing for next step with ReLUs; if a ReLU is layer i,
        // we want to take output shares for the (linear) layer i - 1,
        // and input shares for the (linear) layer i + 1.
        let mut current_layer_shares = Vec::new();
        let mut relu_next_layer_randomizers = Vec::new();
        let relu_time =
            timer_start!(|| format!("ReLU layers offline phase with {} ReLUs", num_relu));
        for &i in &relu_layers {
            let current_layer_output_shares = out_shares
                .get(&(i - 1))
                .expect("should exist because every ReLU should be preceeded by a linear layer");
            current_layer_shares.extend_from_slice(current_layer_output_shares.as_slice().unwrap());

            let next_layer_randomizers = in_shares
                .get(&(i + 1))
                .expect("should exist because every ReLU should be succeeded by a linear layer");
            relu_next_layer_randomizers
                .extend_from_slice(next_layer_randomizers.as_slice().unwrap());
        }

        let crate::gc::ClientState {
            gc_s: relu_circuits,
            server_randomizer_labels: randomizer_labels,
            client_input_labels: relu_labels,
        } = ReluProtocol::<P>::offline_client_protocol(
            &mut reader,
            &mut writer,
            num_relu,
            current_layer_shares.as_slice(),
            rng,
        )?;

        let (relu_client_labels, relu_server_labels) = if num_relu != 0 {
            let size_of_client_input = relu_labels.len() / num_relu;
            let size_of_server_input = randomizer_labels.len() / num_relu;

            assert_eq!(
                size_of_client_input,
                ReluProtocol::<P>::size_of_client_inputs(),
                "number of inputs unequal"
            );

            let client_labels = relu_labels
                .chunks(size_of_client_input)
                .map(|chunk| chunk.to_vec())
                .collect();
            let server_labels = randomizer_labels
                .chunks(size_of_server_input)
                .map(|chunk| chunk.to_vec())
                .collect();

            (client_labels, server_labels)
        } else {
            (vec![], vec![])
        };
        timer_end!(relu_time);

        let approx_time = timer_start!(|| format!(
            "Approx layers offline phase with {} approximations",
            num_approx
        ));
        let approx_state = QuadApproxProtocol::offline_client_protocol::<FPBeaversMul<P>, _, _, _>(
            &mut reader,
            &mut writer,
            &(cfhe_op.unwrap()),
            num_approx,
            rng,
        )?;
        timer_end!(approx_time);
        timer_end!(start_time);
        Ok(ClientState {
            relu_circuits,
            relu_server_labels,
            relu_client_labels,
            relu_next_layer_randomizers,
            approx_state,
            linear_randomizer: in_shares,
            linear_post_application_share: out_shares,
        })
    }

    pub fn online_server_protocol<R: Read + Send, W: Write + Send>(
        mut reader: R,
        mut writer: W,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        state: &ServerState<P>,
    ) -> Result<(), bincode::Error> {
        let (first_layer_in_dims, first_layer_out_dims) = {
            let layer = neural_network.layers.first().unwrap();
            assert!(
                layer.is_linear(),
                "first layer of the network should always be linear."
            );
            (layer.input_dimensions(), layer.output_dimensions())
        };

        let mut num_consumed_relus = 0;
        let mut num_consumed_triples = 0;

        let mut next_layer_input = Output::zeros(first_layer_out_dims);
        let mut next_layer_derandomizer = Input::zeros(first_layer_in_dims);
        let start_time = timer_start!(|| "Server online phase");
        for (i, layer) in neural_network.layers.iter().enumerate() {
            match layer {
                Layer::NLL(NonLinearLayer::ReLU(dims)) => {
                    let start_time = timer_start!(|| "ReLU layer");
                    // Have the server encode the current input, via the garbled circuit,
                    // and then send the labels over to the other party.
                    let layer_size = next_layer_input.len();
                    assert_eq!(dims.input_dimensions(), next_layer_input.dim());
                    let layer_encoders =
                        &state.relu_encoders[num_consumed_relus..(num_consumed_relus + layer_size)];
                    ReluProtocol::online_server_protocol(
                        &mut writer,
                        &next_layer_input.as_slice().unwrap(),
                        layer_encoders,
                    )?;
                    let relu_output_randomizers = state.relu_output_randomizers
                        [num_consumed_relus..(num_consumed_relus + layer_size)]
                        .to_vec();
                    num_consumed_relus += layer_size;
                    next_layer_derandomizer = ndarray::Array1::from_iter(relu_output_randomizers)
                        .into_shape(dims.output_dimensions())
                        .expect("shape should be correct")
                        .into();
                    timer_end!(start_time);
                },
                Layer::NLL(NonLinearLayer::PolyApprox { dims, poly, .. }) => {
                    let start_time = timer_start!(|| "Approx layer");
                    let layer_size = next_layer_input.len();
                    assert_eq!(dims.input_dimensions(), next_layer_input.dim());
                    let triples = &state.approx_state
                        [num_consumed_triples..(num_consumed_triples + layer_size)];
                    num_consumed_triples += layer_size;
                    let shares_of_eval =
                        QuadApproxProtocol::online_server_protocol::<FPBeaversMul<P>, _, _>(
                            SERVER, // party_index: 2
                            &mut reader,
                            &mut writer,
                            &poly,
                            next_layer_input.as_slice().unwrap(),
                            triples,
                        )?;
                    let shares_of_eval: Vec<_> =
                        shares_of_eval.into_iter().map(|s| s.inner.inner).collect();
                    next_layer_derandomizer = ndarray::Array1::from_iter(shares_of_eval)
                        .into_shape(dims.output_dimensions())
                        .expect("shape should be correct")
                        .into();
                    timer_end!(start_time);
                },
                Layer::LL(layer) => {
                    let start_time = timer_start!(|| "Linear layer");
                    // Input for the next layer.
                    let layer_randomizer = state.linear_state.get(&i).unwrap();
                    // The idea here is that the previous layer was linear.
                    // Hence the input we're receiving from the client is
                    if i != 0 && neural_network.layers.get(i - 1).unwrap().is_linear() {
                        next_layer_derandomizer
                            .iter_mut()
                            .zip(&next_layer_input)
                            .for_each(|(l_r, inp)| {
                                *l_r += &inp.inner.inner;
                            });
                    }
                    next_layer_input = Output::zeros(layer.output_dimensions());
                    LinearProtocol::online_server_protocol(
                        &mut reader,
                        layer,
                        layer_randomizer,
                        &next_layer_derandomizer,
                        &mut next_layer_input,
                    )?;
                    next_layer_derandomizer = Output::zeros(layer.output_dimensions());
                    // Since linear operations involve multiplications
                    // by fixed-point constants, we want to truncate here to
                    // ensure that we don't overflow.
                    for share in next_layer_input.iter_mut() {
                        share.inner.signed_reduce_in_place();
                    }
                    timer_end!(start_time);
                },
            }
        }
        timer_end!(start_time);
        let sent_message = MsgSend::new(&next_layer_input);
        bincode::serialize_into(&mut writer, &sent_message)
    }

    /// Outputs shares for the next round's input.
    pub fn online_client_protocol<R: Read + Send, W: Write + Send>(
        mut reader: R,
        mut writer: W,
        input: &Input<FixedPoint<P>>,
        architecture: &NeuralArchitecture<AdditiveShare<P>, FixedPoint<P>>,
        state: &ClientState<P>,
    ) -> Result<Output<FixedPoint<P>>, bincode::Error> {
        let first_layer_in_dims = {
            let layer = architecture.layers.first().unwrap();
            assert!(
                layer.is_linear(),
                "first layer of the network should always be linear."
            );
            assert_eq!(layer.input_dimensions(), input.dim());
            layer.input_dimensions()
        };
        assert_eq!(first_layer_in_dims, input.dim());

        let mut num_consumed_relus = 0;
        let mut num_consumed_triples = 0;

        let start_time = timer_start!(|| "Client online phase");
        let (mut next_layer_input, _) = input.share_with_randomness(&state.linear_randomizer[&0]);

        for (i, layer) in architecture.layers.iter().enumerate() {
            match layer {
                LayerInfo::NLL(dims, nll_info) => {
                    match nll_info {
                        NonLinearLayerInfo::ReLU => {
                            let start_time = timer_start!(|| "ReLU layer");
                            // The client receives the garbled circuits from the server,
                            // uses its already encoded inputs to get the next linear
                            // layer's input.
                            let layer_size = next_layer_input.len();
                            assert_eq!(dims.input_dimensions(), next_layer_input.dim());

                            let layer_client_labels = &state.relu_client_labels
                                [num_consumed_relus..(num_consumed_relus + layer_size)];
                            let layer_server_labels = &state.relu_server_labels
                                [num_consumed_relus..(num_consumed_relus + layer_size)];
                            let next_layer_randomizers = &state.relu_next_layer_randomizers
                                [num_consumed_relus..(num_consumed_relus + layer_size)];

                            let layer_circuits = &state.relu_circuits
                                [num_consumed_relus..(num_consumed_relus + layer_size)];

                            num_consumed_relus += layer_size;

                            let layer_client_labels = layer_client_labels
                                .into_iter()
                                .flat_map(|l| l.clone())
                                .collect::<Vec<_>>();
                            let layer_server_labels = layer_server_labels
                                .into_iter()
                                .flat_map(|l| l.clone())
                                .collect::<Vec<_>>();
                            let output = ReluProtocol::online_client_protocol(
                                &mut reader,
                                layer_size,              // num_relus
                                &layer_server_labels,    // Labels for layer
                                &layer_client_labels,    // Labels for layer
                                &layer_circuits,         // circuits for layer.
                                &next_layer_randomizers, // circuits for layer.
                            )?;
                            next_layer_input = ndarray::Array1::from_iter(output)
                                .into_shape(dims.output_dimensions())
                                .expect("shape should be correct")
                                .into();
                            timer_end!(start_time);
                        },
                        NonLinearLayerInfo::PolyApprox { poly, .. } => {
                            let start_time = timer_start!(|| "Approx layer");
                            let layer_size = next_layer_input.len();
                            assert_eq!(dims.input_dimensions(), next_layer_input.dim());
                            let triples = &state.approx_state
                                [num_consumed_triples..(num_consumed_triples + layer_size)];
                            num_consumed_triples += layer_size;
                            let output = QuadApproxProtocol::online_client_protocol::<
                                FPBeaversMul<P>,
                                _,
                                _,
                            >(
                                CLIENT, // party_index: 1
                                &mut reader,
                                &mut writer,
                                &poly,
                                next_layer_input.as_slice().unwrap(),
                                triples,
                            )?;
                            next_layer_input = ndarray::Array1::from_iter(output)
                                .into_shape(dims.output_dimensions())
                                .expect("shape should be correct")
                                .into();
                            next_layer_input
                                .randomize_local_share(&state.linear_randomizer[&(i + 1)]);
                            timer_end!(start_time);
                        },
                    }
                },
                LayerInfo::LL(_, layer_info) => {
                    let start_time = timer_start!(|| "Linear layer");
                    // Send server secret share if required by the layer
                    let input = next_layer_input;
                    next_layer_input = state.linear_post_application_share[&i].clone();

                    LinearProtocol::online_client_protocol(
                        &mut writer,
                        &input,
                        &layer_info,
                        &mut next_layer_input,
                    )?;
                    // If this is not the last layer, and if the next layer
                    // is also linear, randomize the output correctly.
                    if i != (architecture.layers.len() - 1)
                        && architecture.layers[i + 1].is_linear()
                    {
                        next_layer_input.randomize_local_share(&state.linear_randomizer[&(i + 1)]);
                    }
                    timer_end!(start_time);
                },
            }
        }

        timer_end!(start_time);
        bincode::deserialize_from(reader).map(|output: MsgRcv<P>| {
            let server_output_share = output.msg();
            server_output_share.combine(&next_layer_input)
        })
    }
}
