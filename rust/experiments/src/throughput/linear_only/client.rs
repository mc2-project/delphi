use experiments::linear_only::construct_networks;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

fn main() {
    let num_clients = std::env::args()
        .nth(1)
        .expect("Please pass number of clients as input.")
        .parse()
        .expect("should be positive integer"); // Get number of polys
    let batch_size = std::env::args()
        .nth(2)
        .expect("Please pass batch size as input.")
        .parse()
        .expect("should be positive integer"); // Get number of polys
    let use_cuda: u32 = std::env::args()
        .nth(3)
        .expect("Please indicate whether CUDA should be used.")
        .parse()
        .expect("should be positive integer"); // Get number of polys
    let use_cuda = use_cuda == 0;
    let vs = if use_cuda {
        tch::nn::VarStore::new(tch::Device::cuda_if_available())
    } else {
        tch::nn::VarStore::new(tch::Device::Cpu)
    };

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let network = construct_networks(Some(&vs.root()), batch_size, &mut rng);
    let architectures = network
        .into_iter()
        .map(|(i, n)| (i, (&n).into()))
        .collect::<Vec<_>>();

    let server_addr = "127.0.0.1:8002";
    experiments::throughput::client::nn_client(num_clients, server_addr, &architectures, &mut rng);
}
