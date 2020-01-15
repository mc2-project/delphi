#!/bin/bash

resnet_layers=( 0 6 12 14 16 18 20 22 24 26 )
minionn_layers=( 0 1 2 3 5 6 7 )

mkdir -p latency_data
err="latency_data/err.log"

for i in {0..4}
do
  mkdir -p latency_data/run$i
  for l in "${resnet_layers[@]}"
  do
    echo "ResNet32 with $l layers on 4 threads - Run #$i"
    run_path="latency_data/run$i/resnet32-4-$l.txt"
    env RAYON_NUM_THREADS=4 CLICOLOR=0 cargo +nightly run --all-features --release --bin resnet32-client $l 2>$err > $run_path;
    cat "$run_path/resnet-32-4-$l.txt" | egrep "End.*Client online|End.*offline phase|End.*ReLU layer\."
    echo -e "\n"
    sleep 2
  done
done

for i in {0..4}
do
  for l in "${minionn_layers[@]}"
  do
    echo "MiniONN with $l layers on 4 threads - Run #$i"
    run_path="latency_data/run$i/minionn-4-$l.txt"
    env RAYON_NUM_THREADS=4 CLICOLOR=0 cargo +nightly run --all-features --release --bin minionn-client $l 2>$err > $run_path;
    cat "$run_path/minionn-4-$l.txt" | egrep "End.*Client online|End.*offline phase|End.*ReLU layer\."
    echo -e "\n"
    sleep 2
  done
done
