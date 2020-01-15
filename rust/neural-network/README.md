# Instructions for running GPU-based benchmarks:

```
$ ssh <username>@c75.millennium.berkeley.edu
$ export HOME=/data/<username>
$ cd delphi/system/rust/neural-networks/benches/
$ RUSTFLAGS="-C link-args=-Wl,-rpath,/usr/lib/x86_64-linux-gnu/libgomp.so.1" cargo bench test_torch_
```
