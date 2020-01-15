#!/bin/bash
resnet_layers=( 0 6 12 14 16 18 20 22 24 26 )
minionn_layers=( 0 1 2 3 5 6 7 )

for j in {0..4}
do
  for i in "${resnet_layers[@]}"
  do
    env RAYON_NUM_THREADS=4 cargo +nightly run --all-features --release --bin resnet32-server $i
  done;
done

for j in {0..4}
do
  for i in "${minionn_layers[@]}"
  do
    env RAYON_NUM_THREADS=4 cargo +nightly run --all-features --release --bin minionn-server $i
  done
done
