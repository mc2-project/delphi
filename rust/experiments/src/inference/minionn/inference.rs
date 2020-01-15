use experiments::minionn::construct_minionn;
use neural_network::{
    npy::NpyData,
    ndarray::Array4,
};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use std::{
    path::Path,
    io::Read,
};

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4,
    0x76, 0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a,
    0x52, 0xd2,
];


fn main() {
    let weights_path = std::env::args().nth(1).expect("Pass weights path as first argument");
    let num_polys = std::env::args().nth(2).expect("Pass number of polynomial layers as second argument").parse().expect("should be positive integer");

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    // Build network 
    let mut network = construct_minionn(None, 1, num_polys, &mut rng);
    let architecture = (&network).into();
    
    // Load network weights
    network.from_numpy(&weights_path).unwrap();

    // Open image and class
    let mut buf = vec![];
    std::fs::File::open(Path::new("class.npy")).unwrap()
        .read_to_end(&mut buf).unwrap();
    let class: i64 = NpyData::from_bytes(&buf).unwrap().to_vec()[0];

    buf = vec![];
    std::fs::File::open(Path::new("image.npy")).unwrap()
        .read_to_end(&mut buf).unwrap();
    let image_vec: Vec<f64> = NpyData::from_bytes(&buf).unwrap().to_vec();
    let image = Array4::from_shape_vec((1, 3, 32, 32), image_vec).unwrap(); 

    experiments::inference::inference::run(network, architecture, image, class);
}
