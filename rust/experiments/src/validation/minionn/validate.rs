use clap::{App, Arg, ArgMatches};
use experiments::minionn::construct_minionn;
use neural_network::{ndarray::Array4, npy::NpyData};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use std::{io::Read, path::Path};

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

fn get_args() -> ArgMatches<'static> {
    App::new("minionn-accuracy")
        .arg(
            Arg::with_name("weights")
                .short("w")
                .long("weights")
                .takes_value(true)
                .help("Path to weights")
                .required(true),
        )
        .arg(
            Arg::with_name("images")
                .short("i")
                .long("images")
                .takes_value(true)
                .help("Path to test images")
                .required(true),
        )
        .arg(
            Arg::with_name("layers")
                .short("l")
                .long("layers")
                .takes_value(true)
                .help("Number of polynomial layers (0-7)")
                .required(true),
        )
        .get_matches()
}

fn main() {
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let args = get_args();
    let weights = args.value_of("weights").unwrap();
    let images = args.value_of("images").unwrap();
    let layers = clap::value_t!(args.value_of("layers"), usize).unwrap();

    // Build network
    let mut network = construct_minionn(None, 1, layers, &mut rng);
    let architecture = (&network).into();

    // Load network weights
    network.from_numpy(&weights).unwrap();

    // Open all images, classes, and classification results
    let data_dir = Path::new(&images);

    let mut buf = vec![];
    std::fs::File::open(data_dir.join(Path::new("classes.npy")))
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let classes: Vec<i64> = NpyData::from_bytes(&buf).unwrap().to_vec();

    buf = vec![];
    std::fs::File::open(data_dir.join(Path::new("plaintext.npy")))
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let plaintext: Vec<i64> = NpyData::from_bytes(&buf).unwrap().to_vec();

    let mut images: Vec<Array4<f64>> = Vec::new();
    for i in 0..classes.len() {
        buf = vec![];
        std::fs::File::open(data_dir.join(Path::new(&format!("image_{}.npy", i))))
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        let image_vec: Vec<f64> = NpyData::from_bytes(&buf).unwrap().to_vec();
        let input = Array4::from_shape_vec((1, 3, 32, 32), image_vec).unwrap();
        images.push(input);
    }
    experiments::validation::validate::run(network, architecture, images, classes, plaintext);
}
