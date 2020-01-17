<h1 align="center">Delphi</h1>

___Delphi___ is a Python, C++, and Rust library for **Secure Deep Neural Network Inference**

This library was initially developed as part of the paper *"[Delphi: A Cryptographic Inference Service for Neural Networks][delphi]"*, and is released under the MIT License and the Apache v2 License (see [License](#license)).

**WARNING:** This is an academic proof-of-concept prototype, and in particular has not received careful code review. This implementation is NOT ready for production use.

## Overview

This library implements a cryptographic system for efficient inference on general convolutional neural networks.

The construction utilizes an array of multi-party computation and machine-learning techniques, as described in the [Delphi paper][delphi].

## Directory structure

This repository contains several folders that implement the different building blocks of Delphi. The high-level structure of the repository is as follows.
* [`python`](python): Example Python scripts for performing neural architecture search (NAS)

* [`rust/algebra`](rust/algebra): Rust crate that provides finite fields

* [`rust/crypto-primitives`](rust/crypto-primitives): Rust crate that implements some useful cryptographic primitives

* [`rust/experiments`](rust/experiments): Rust crate for running latency, bandwidth, throughput, accuracy, and memory usage experiments

* [`rust/neural-network`](rust/neural-network): Rust crate that implements generic neural networks

* [`rust/protocols`](rust/protocols): Rust crate that implements cryptographic protocols

* [`rust/protocols-sys`](rust/crypto-primitives): Rust crate that provides the C++ backend for Delphi's pre-processing phase and an FFI for the backend

In addition, there is a  [`rust/bench-utils`](rust/bench-utils) crate which contains infrastructure for benchmarking. This crate includes macros for timing code segments and is used for profiling the building blocks of Delphi.


## Build guide

The library compiles on the `nightly` toolchain of the Rust compiler. To install the latest version of Rust, first install `rustup` by following the instructions [here](https://rustup.rs/), or via your platform's package manager. Once `rustup` is installed, install the Rust toolchain by invoking:
```bash
rustup install nightly
```

After that, use `cargo`, the standard Rust build tool, to build the library:
```bash
git clone https://github.com/mc2-project/delphi
cd delphi
cargo +nightly build --release
```

This library comes with unit tests for each of the provided crates. Run the tests with:
```bash
cargo +nightly test
``` 
Benchmarks are included for the following crates:
- [`algebra`](algebra)
- [`crypto-primitives`](algebra)
- [`neural-network`](algebra)

Run the benchmarks with:
```bash
cargo +nightly bench
```

## End-to-End Example 
Let's walk through a full example of using Delphi with the pretrained MiniONN model at `python/minionn/pretrained/relu/model`.

Using Python3.6/7, install the following packages:
```bash
pip install ray[tune]==0.8.0 requests scipy tensorflow==1.15.0 
```
Optionally, visualizing results requires the `tensorboard` package:
```bash
pip install tensorboard==1.15.0
```
### Model Preprocessing
For this step we will be working exclusively in the `python/` subdirectory:
```bash
cd python/
```
#### 1. Neural Architecture Search
In order to achieve optimal performance, we need to run NAS on our pretrained model to optimize which activation layers to approximate (see [Delphi paper][delphi] for more details).

Start by running NAS on the total number of activation layers (`-a` flag), in this case 7, with the following command: 
```bash
python minionn/minionn_cifar10.py -n 7_layers -d /tmp -a 7 -r minionn/pretrained/relu/model 
```
After that has finished, we can review the results by running:
```bash
tensorboard --logdir=/tmp/7_layers
```
and visiting `localhost:6006` in a web browser.

 If the resulting networks don't have satisfactory accuracy, rerun the above command with a lower number of activation layers.

#### 2. Serialization

Once a satisfactory network is trained (say at `/tmp/model`), we need to extract the model weights for use in the Rust cryptographic protocol. This can be done by running the following command.
```bash
python extract_keras_weights.py 0 -w /tmp/model -a {approx layers}
```
Where `{approx layers}` is the list of layers being approximated. 

This will output a `model.npy` file which is ready to be used by Delphi for secure inference.

Passing the `-q` flag additionally quantizes the model for additional performance, but may lower accuracy. Additionally passing the `-t` flag will test accuracy of the original and quantized network in order for you to decide which is best. Currently, quantizing requires Tensorflow 2.0.

### Performing inference with the model
We will now use this model to perform inference on an image using the code in `rust/experiments/src/inference`:
```bash
cd ../rust/experiments/src/inference
```
Run the following python script to generate a test image:
```bash
python3 generate_test_images.py
```
and perform inference on that image by running:
```bash
cargo +nightly run --release --bin minionn-inference /tmp/model.npy {num of approx layers}
```

## License

Delphi is licensed under either of the following licenses, at your discretion.

 * Apache License Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

Unless you explicitly state otherwise, any contribution submitted for inclusion in Delphi by you shall be dual licensed as above (as defined in the Apache v2 License), without any additional terms or conditions.

[delphi]: https://eprint.iacr.org/2020/050.pdf

## Reference paper

[_Delphi: A Cryptographic Inference Service for Neural Networks_][delphi]    
[Pratyush Mishra](https://www.github.com/pratyush), [Ryan Lehmkuhl](https://www.github.com/ryanleh), Akshayaram Srinivasan, Wenting Zheng, and  Raluca Ada Popa    
*Usenix Security Symposium 2020*

## Acknowledgements

This work was supported by:
the National Science Foundation;
and donations from Sloan Foundation, Bakar and Hellman Fellows Fund, Alibaba, Amazon Web Services, Ant Financial, Arm, Capital One, Ericsson, Facebook, Google, Intel, Microsoft, Scotiabank, Splunk and VMware

Some parts of the finite field arithmetic infrastructure in the `algebra` crate have been adapted from code in the [`algebra`](https://github.com/scipr-lab/zexe) crate.
