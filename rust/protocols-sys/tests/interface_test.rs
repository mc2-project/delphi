use algebra::{
    fields::{near_mersenne_64::F, PrimeField},
    fixed_point::{FixedPoint, FixedPointParameters},
    UniformRandom,
};
use crypto_primitives::{additive_share::Share, beavers_mul::Triple};
use itertools::izip;
use ndarray::s;
use neural_network::{layers::*, tensors::*, Evaluate};
use protocols_sys::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;

type AdditiveShare<P> = crypto_primitives::AdditiveShare<FixedPoint<P>>;

struct TenBitExpParams {}
impl FixedPointParameters for TenBitExpParams {
    type Field = F;
    const MANTISSA_CAPACITY: u8 = 3;
    const EXPONENT_CAPACITY: u8 = 8;
}

type TenBitExpFP = FixedPoint<TenBitExpParams>;
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

fn print_output_f64(output: &Output<TenBitExpFP>) {
    output
        .slice(s![0, .., .., ..])
        .outer_iter()
        .for_each(|out_c| {
            out_c.outer_iter().for_each(|inp_c| {
                inp_c.iter().for_each(|e| print!("{:.2}, ", f64::from(*e)));
                println!("");
            });
        });
}

fn print_output_u64(output: &Output<TenBitExpFP>) {
    output
        .slice(s![0, .., .., ..])
        .outer_iter()
        .for_each(|out_c| {
            out_c.outer_iter().for_each(|inp_c| {
                inp_c
                    .iter()
                    .for_each(|e| print!("{:.2}, ", e.inner.into_repr().0));
                println!("");
            });
        });
}

// Compares floats to 2 decimal points
fn approx_equal(f1: f64, f2: f64) -> bool {
    f64::trunc(100. * f1) == f64::trunc(100. * f2)
}

fn interface<R: Rng + rand::CryptoRng>(
    client_cg: &mut SealClientCG,
    server_cg: &mut SealServerCG,
    input_dims: (usize, usize, usize, usize),
    output_dims: (usize, usize, usize, usize),
    pt_layer: &LinearLayer<TenBitExpFP, TenBitExpFP>,
    rng: &mut R,
) -> bool {
    // Client preprocessing
    let mut r = Input::zeros(input_dims);
    r.iter_mut()
        .for_each(|e| *e = generate_random_number(rng).1);

    let input_ct_vec = client_cg.preprocess(&r.to_repr());

    // Server preprocessing
    let mut linear_share = Output::zeros(output_dims);
    linear_share
        .iter_mut()
        .for_each(|e| *e = generate_random_number(rng).1);

    server_cg.preprocess(&linear_share.to_repr());

    // Server receive ciphertext and compute convolution
    let linear_ct = server_cg.process(input_ct_vec);

    // Client receives ciphertexts
    client_cg.decrypt(linear_ct);

    // The interface changed here after making this test so from here on
    // the code is very messy
    let mut linear = Output::zeros(output_dims);
    client_cg.postprocess::<TenBitExpParams>(&mut linear);

    let mut success = true;

    println!("\nPlaintext linear:");
    let linear_pt = pt_layer.evaluate(&r);
    print_output_f64(&linear_pt);

    println!("Linear:");
    let mut linear_result = Output::zeros(output_dims);
    linear_result
        .iter_mut()
        .zip(linear.iter().zip(linear_share.iter()))
        .zip(linear_pt.iter())
        .for_each(|((r, (s1, s2)), p)| {
            *r = FixedPoint::randomize_local_share(s1, &s2.inner).inner;
            success &= approx_equal(f64::from(*r), f64::from(*p));
        });
    print_output_f64(&linear_result);
    success
}

#[test]
fn test_convolution() {
    use neural_network::layers::convolution::*;

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    // Set the parameters for the convolution.
    let input_dims = (1, 1, 28, 28);
    let kernel_dims = (16, 1, 5, 5);
    let stride = 1;
    let padding = Padding::Valid;
    // Sample a random kernel.
    let mut kernel = Kernel::zeros(kernel_dims);
    let mut bias = Kernel::zeros((kernel_dims.0, 1, 1, 1));
    kernel
        .iter_mut()
        .for_each(|ker_i| *ker_i = generate_random_number(&mut rng).1);
    // Offline phase doesn't interact with bias so this can be 0
    bias.iter_mut()
        .for_each(|bias_i| *bias_i = TenBitExpFP::from(0.0));

    let layer_params =
        Conv2dParams::<TenBitAS, _>::new(padding, stride, kernel.clone(), bias.clone());
    let pt_layer_params =
        Conv2dParams::<TenBitExpFP, _>::new(padding, stride, kernel.clone(), bias.clone());
    let output_dims = layer_params.calculate_output_size(input_dims);
    let layer_dims = LayerDims {
        input_dims,
        output_dims,
    };
    let layer = LinearLayer::Conv2d {
        dims: layer_dims,
        params: layer_params,
    };
    let layer_info = (&layer).into();

    let pt_layer = LinearLayer::Conv2d {
        dims: layer_dims,
        params: pt_layer_params,
    };

    // Keygen
    let mut key_share = KeyShare::new();
    let (cfhe, keys_vec) = key_share.generate();
    let sfhe = key_share.receive(keys_vec);

    let input_dims = layer.input_dimensions();
    let output_dims = layer.output_dimensions();

    let mut client_cg = SealClientCG::Conv2D(client_cg::Conv2D::new(
        &cfhe,
        &layer_info,
        input_dims,
        output_dims,
    ));
    let mut server_cg =
        SealServerCG::Conv2D(server_cg::Conv2D::new(&sfhe, &layer, &kernel.to_repr()));

    assert_eq!(
        interface(
            &mut client_cg,
            &mut server_cg,
            input_dims,
            output_dims,
            &pt_layer,
            &mut rng
        ),
        true
    );
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
        .for_each(|ker_i| *ker_i = generate_random_number(&mut rng).1);
    // Offline phase doesn't interact with bias so this can be 0
    bias.iter_mut()
        .for_each(|bias_i| *bias_i = TenBitExpFP::from(0.0));

    let layer_params = FullyConnectedParams::<TenBitAS, _>::new(kernel.clone(), bias.clone());
    let pt_layer_params = FullyConnectedParams::<TenBitExpFP, _>::new(kernel.clone(), bias.clone());
    let output_dims = layer_params.calculate_output_size(input_dims);
    let layer_dims = LayerDims {
        input_dims,
        output_dims,
    };
    let layer = LinearLayer::FullyConnected {
        dims: layer_dims,
        params: layer_params,
    };
    let layer_info = (&layer).into();

    let pt_layer = LinearLayer::FullyConnected {
        dims: layer_dims,
        params: pt_layer_params,
    };

    // Keygen
    let mut key_share = KeyShare::new();
    let (cfhe, keys_vec) = key_share.generate();
    let sfhe = key_share.receive(keys_vec);

    let input_dims = layer.input_dimensions();
    let output_dims = layer.output_dimensions();

    let mut client_cg = SealClientCG::FullyConnected(client_cg::FullyConnected::new(
        &cfhe,
        &layer_info,
        input_dims,
        output_dims,
    ));
    let mut server_cg = SealServerCG::FullyConnected(server_cg::FullyConnected::new(
        &sfhe,
        &layer,
        &kernel.to_repr(),
    ));

    assert_eq!(
        interface(
            &mut client_cg,
            &mut server_cg,
            input_dims,
            output_dims,
            &pt_layer,
            &mut rng
        ),
        true
    );
}

#[inline]
fn to_u64(x: &Vec<F>) -> Vec<u64> {
    x.iter().map(|e| e.into_repr().0).collect()
}

#[test]
fn test_triple_gen() {
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    let num = 100000;

    // Keygen
    let mut key_share = KeyShare::new();
    let (cfhe, keys_vec) = key_share.generate();
    let sfhe = key_share.receive(keys_vec);

    let mut client_gen = SealClientGen::new(&cfhe);
    let mut server_gen = SealServerGen::new(&sfhe);

    let mut client_a = Vec::with_capacity(num);
    let mut client_b = Vec::with_capacity(num);

    let mut server_a = Vec::with_capacity(num);
    let mut server_b = Vec::with_capacity(num);
    let mut server_c = Vec::with_capacity(num);
    let mut server_r = Vec::with_capacity(num);
    for i in 0..num {
        client_a.push(F::uniform(&mut rng));
        client_b.push(F::uniform(&mut rng));

        server_a.push(F::uniform(&mut rng));
        server_b.push(F::uniform(&mut rng));
        server_c.push(F::uniform(&mut rng));
        server_r.push(server_a[i] * server_b[i] - server_c[i]);
    }

    let (mut client_triples, mut a_ct, mut b_ct) =
        client_gen.triples_preprocess(to_u64(&client_a).as_slice(), to_u64(&client_b).as_slice());

    let mut server_triples = server_gen.triples_preprocess(
        to_u64(&server_a).as_slice(),
        to_u64(&server_b).as_slice(),
        to_u64(&server_r).as_slice(),
    );

    let mut c_ct = server_gen.triples_online(
        &mut server_triples,
        a_ct.as_mut_slice(),
        b_ct.as_mut_slice(),
    );

    let client_c = client_gen.triples_postprocess(&mut client_triples, c_ct.as_mut_slice());

    let server_triples = izip!(server_a, server_b, server_c,).map(|(a, b, c)| Triple { a, b, c });

    let client_triples = izip!(client_a, client_b, client_c,).map(|(a, b, c)| Triple {
        a: F::from_repr(a.into()),
        b: F::from_repr(b.into()),
        c: F::from_repr(c.into()),
    });

    izip!(server_triples, client_triples).for_each(|(s, c)| {
        let a = s.a + &c.a;
        let b = s.b + &c.b;
        let c = s.c + &c.c;
        assert_eq!(c, a * b);
    });
}
