use clap::{App, Arg, ArgMatches};
use experiments::linear_only::construct_networks;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

fn get_args() -> ArgMatches<'static> {
    App::new("tp-client")
        .arg(
            Arg::with_name("ip")
                .short("i")
                .long("ip")
                .takes_value(true)
                .help("Server IP address")
                .required(true),
        )
        .arg(
            Arg::with_name("clients")
                .short("c")
                .long("clients")
                .takes_value(true)
                .help("Number of clients")
                .required(true),
        )
        .arg(
            Arg::with_name("batch")
                .short("b")
                .long("batch")
                .takes_value(true)
                .help("Batch size")
                .required(true),
        )
        .arg(
            Arg::with_name("gpu")
                .short("g")
                .long("gpu")
                .help("Whether to use a GPU (0/1)"),
        )
        .arg(
            Arg::with_name("port")
                .short("p")
                .long("port")
                .takes_value(true)
                .help("Server port (default 8000)")
                .required(false),
        )
        .get_matches()
}

fn main() {
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let args = get_args();
    let ip = args.value_of("ip").unwrap();
    let num_clients = clap::value_t!(args.value_of("clients"), usize).unwrap();
    let batch_size = clap::value_t!(args.value_of("batch"), usize).unwrap();
    let use_gpu = args.is_present("gpu");
    let port = args.value_of("port").unwrap_or("8000");
    let server_addr = format!("{}:{}", ip, port);

    let vs = if use_gpu {
        tch::nn::VarStore::new(tch::Device::cuda_if_available())
    } else {
        tch::nn::VarStore::new(tch::Device::Cpu)
    };

    let network = construct_networks(Some(&vs.root()), batch_size, &mut rng);
    let architectures = network
        .into_iter()
        .map(|(i, n)| (i, (&n).into()))
        .collect::<Vec<_>>();

    experiments::throughput::client::nn_client(num_clients, &server_addr, &architectures, &mut rng);
}
