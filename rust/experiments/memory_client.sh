#! /bin/bash
# Run:
#  $ grep mem_heap_B server.out | sed -e 's/mem_heap_B=\(.*\)/\1/' | sort -g | tail -n 1
# To just output the peak memory usage in bytes

resnet_layers=( 0 6 12 14 16 18 20 22 24 26 )
minionn_layers=( 3 5 6 7 )

mkdir -p memory_data
err="memory_data/err.log"

for l in "${resnet_layers[@]}"
do
 path="memory_data/resnet32-client-$l.out"
 valgrind --tool=massif --pages-as-heap=no --massif-out-file=$path ../target/release/resnet32-client $l
 sleep 10
done

for l in "${minionn_layers[@]}"
do
 path="memory_data/minionn-client-$l.out"
 valgrind --tool=massif --pages-as-heap=no --massif-out-file=$path ../target/release/minionn-client $l
 sleep 10
done
