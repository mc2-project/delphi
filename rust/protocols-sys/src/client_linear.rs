use crate::{
    client_linear::SEALClientLinear::{Conv2d, FullyConnected},
    *,
};
use algebra::{fixed_point::*, fp_64::Fp64Parameters, FpParameters, PrimeField};
use neural_network::{
    layers::{convolution::Padding, LinearLayerInfo},
    tensors::Input,
};
use std::{os::raw::c_char, ptr::null_mut};

pub enum SEALClientLinear<'a> {
    Conv2d {
        data:   Metadata,
        cfhe:   &'a ClientFHE,
        ct:     *mut c_char,
        result: *mut *mut u64,
    },
    FullyConnected {
        data:   Metadata,
        cfhe:   &'a ClientFHE,
        ct:     *mut c_char,
        result: *mut u64,
    },
}

impl<'a> SEALClientLinear<'a> {
    pub fn new<F, C>(
        cfhe: &'a ClientFHE,
        layer_info: &LinearLayerInfo<F, C>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
    ) -> Self {
        match layer_info {
            LinearLayerInfo::Conv2d {
                kernel,
                padding,
                stride,
            } => {
                let data = unsafe {
                    conv_metadata(
                        cfhe.encoder,
                        input_dims.2 as i32,
                        input_dims.3 as i32,
                        kernel.2 as i32,
                        kernel.3 as i32,
                        kernel.1 as i32,
                        kernel.0 as i32,
                        *stride as i32,
                        *stride as i32,
                        *padding == Padding::Valid,
                        2,
                    )
                };
                Conv2d {
                    data,
                    cfhe,
                    ct: null_mut(),
                    result: null_mut(),
                }
            },
            LinearLayerInfo::FullyConnected => {
                let data = unsafe {
                    fc_metadata(
                        cfhe.encoder,
                        (input_dims.1 * input_dims.2 * input_dims.3) as i32,
                        output_dims.1 as i32,
                    )
                };
                FullyConnected {
                    data,
                    cfhe,
                    ct: null_mut(),
                    result: null_mut(),
                }
            },
            _ => panic!("Identity/AvgPool layers do not use SEAL in the offline phase"),
        }
    }

    pub fn preprocess<F, P>(&mut self, client_rand: &Input<F>) -> Vec<c_char>
    where
        F: Copy + Into<FixedPoint<P>>,
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
    {
        // Convert the client share from AdditiveShare -> u64
        let mut client_rand_c = Input::zeros(client_rand.dim());
        client_rand_c
            .iter_mut()
            .zip(client_rand.iter())
            .for_each(|(e1, e2)| {
                let fp: FixedPoint<P> = (*e2).into();
                *e1 = fp.inner.into_repr().0;
            });

        match self {
            Conv2d {
                data,
                cfhe,
                ct,
                result: _,
            } => {
                // Convert client secret share to raw pointers for C FFI
                let client_rand_vec: Vec<*const u64> = (0..data.inp_chans)
                    .into_iter()
                    .map(|inp_c| {
                        client_rand_c
                            .slice(s![0, inp_c, .., ..])
                            .as_slice()
                            .expect("Error converting client share")
                            .as_ptr()
                    })
                    .collect();
                let mut enc_size: u64 = 0;
                unsafe {
                    *ct = client_conv_preprocess(
                        client_rand_vec.as_ptr(),
                        *cfhe,
                        data,
                        2,
                        &mut enc_size,
                    );
                    std::slice::from_raw_parts(*ct, enc_size as usize).to_vec()
                }
            },
            FullyConnected {
                data,
                cfhe,
                ct,
                result: _,
            } => {
                // Convert client secret share to raw pointers for C FFI
                let client_rand_vec: *const u64 = client_rand_c
                    .slice(s![0, .., .., ..])
                    .as_slice()
                    .expect("Error converting client share")
                    .as_ptr();
                let mut enc_size: u64 = 0;
                unsafe {
                    *ct = client_fc_preprocess(client_rand_vec, *cfhe, data, &mut enc_size);
                    std::slice::from_raw_parts(*ct, enc_size as usize).to_vec()
                }
            },
        }
    }

    pub fn decrypt(&mut self, enc_result: *mut c_char) {
        match self {
            Conv2d {
                data,
                cfhe,
                ct: _,
                result,
            } => unsafe {
                *result = client_conv_decrypt(enc_result, *cfhe, data);
            },
            FullyConnected {
                data,
                cfhe,
                ct: _,
                result,
            } => unsafe {
                *result = client_fc_decrypt(enc_result, *cfhe, data);
            },
        }
    }

    pub fn postprocess<F, P>(&self, client_share_next: &mut Input<F>)
    where
        F: From<FixedPoint<P>>,
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
    {
        match self {
            Conv2d {
                data,
                cfhe: _,
                ct: _,
                result,
            } => {
                for chan in 0..data.out_chans as usize {
                    for row in 0..data.output_h as usize {
                        for col in 0..data.output_w as usize {
                            let idx = (row * (data.output_w as usize) + col) as isize;
                            let reduced = unsafe {
                                *(*(result.offset(chan as isize))).offset(idx as isize)
                                    % <P::Field as PrimeField>::Params::MODULUS.0
                            };
                            client_share_next[[0, chan, row, col]] =
                                FixedPoint::with_num_muls(P::Field::from_repr(reduced.into()), 1)
                                    .into();
                        }
                    }
                }
            },
            FullyConnected {
                data,
                cfhe: _,
                ct: _,
                result,
            } => {
                for row in 0..data.filter_h as usize {
                    let reduced = unsafe {
                        *(result.offset(row as isize)) % <P::Field as PrimeField>::Params::MODULUS.0
                    };
                    client_share_next[[0, row, 0, 0]] =
                        FixedPoint::with_num_muls(P::Field::from_repr(reduced.into()), 1).into();
                }
            },
        }
    }
}

impl<'a> Drop for SEALClientLinear<'a> {
    fn drop(&mut self) {
        match self {
            Conv2d {
                data,
                cfhe: _,
                ct,
                result,
            } => unsafe { client_conv_free(data, *ct, *result, 2) },
            FullyConnected {
                data: _,
                cfhe: _,
                ct,
                result,
            } => unsafe {
                client_fc_free(*ct, *result);
            },
        }
    }
}
