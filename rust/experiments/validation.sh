#!/bin/bash

weights=/home/ryan/research/delphi/system/python/minionn/pretrained/7_approx/model_quant.npy
images=/home/ryan/research/delphi/system/rust/experiments/src/validation/
poly=7
params='8;3'

mkdir -p validation_data
err="validation_data/err.log"

for tuple in $params;
do 
    exp=$(echo $tuple | cut -d ';' -f 1);
    man=$(echo $tuple | cut -d ';' -f 2);
    data_path=validation_data/params_$exp\_$man
    echo -e "\n\n-----------------------------------------";
    echo "EXPONENT: $exp, MANTISSA: $man";
    echo "-----------------------------------------";
    sed -i -e "s/\(.*const EXPONENT_CAPACITY: u8 =.\)\([0-9]\+\)/\1$exp/g" src/lib.rs;
    sed -i -e "s/\(.*const MANTISSA_CAPACITY: u8 =.\)\([0-9]\+\)/\1$man/g" src/lib.rs;
    cargo +nightly run --bin minionn-accuracy --release -- $weights $images $poly --nocapture 2>$err | tee $data_path;
done
