#! /bin/bash

resnet_layers=( 0 6 12 14 16 18 20 22 24 26 )
resnet_layers=(26)

mkdir -p bandwidth_data
err="bandwidth_data/err.log"

for l in "${resnet_layers[@]}"
do
 path="bandwidth_data/resnet32-$l.pcap"
 tshark -i ens5 -w $path &
 sleep 5
 pid=$!
 cargo +nightly run --all-features --release --bin resnet32-client $l;
 sudo kill $pid
done

for l in "${minionn_layers[@]}"
do
 path="bandwidth_data/minionn-$l.pcap"
 tshark -i ens5 -w $path &
 sleep 5
 pid=$!
 cargo +nightly run --all-features --release --bin minionn-client $l;
 sudo kill $pid
done
