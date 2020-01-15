use crate::{
    server_linear::SEALServerLinear::{Conv2d, FullyConnected},
    *,
};
use neural_network::{
    layers::{convolution::Padding, LinearLayer},
    tensors::{Kernel, Output},
};
use std::{os::raw::c_char, ptr::null_mut};

pub enum SEALServerLinear<'a> {
    Conv2d {
        data:   Metadata,
        sfhe:   &'a ServerFHE,
        masks:  *mut *mut *mut *mut c_char,
        noise:  *mut *mut c_char,
        result: *mut c_char,
    },
    FullyConnected {
        data:   Metadata,
        sfhe:   &'a ServerFHE,
        masks:  *mut *mut c_char,
        noise:  *mut c_char,
        result: *mut c_char,
    },
}

impl<'a> SEALServerLinear<'a> {
    pub fn new<F, C>(sfhe: &'a ServerFHE, layer: &LinearLayer<F, C>) -> Self {
        let (input_dims, output_dims, kernel_dims) = layer.all_dimensions();
        match layer {
            LinearLayer::Conv2d { params, .. } => {
                let data = unsafe {
                    conv_metadata(
                        sfhe.encoder,
                        input_dims.2 as i32,
                        input_dims.3 as i32,
                        kernel_dims.2 as i32,
                        kernel_dims.3 as i32,
                        kernel_dims.1 as i32,
                        kernel_dims.0 as i32,
                        params.stride as i32,
                        params.stride as i32,
                        params.padding == Padding::Valid,
                        2,
                    )
                };
                Conv2d {
                    data,
                    sfhe,
                    masks: null_mut(),
                    noise: null_mut(),
                    result: null_mut(),
                }
            },
            LinearLayer::FullyConnected { .. } => {
                let data = unsafe {
                    fc_metadata(
                        sfhe.encoder,
                        (input_dims.1 * input_dims.2 * input_dims.3) as i32,
                        output_dims.1 as i32,
                    )
                };
                FullyConnected {
                    data,
                    sfhe,
                    masks: null_mut(),
                    noise: null_mut(),
                    result: null_mut(),
                }
            },
            _ => panic!("Identity/AvgPool layers do not use SEAL in the offline phase"),
        }
    }

    pub fn preprocess(&mut self, server_randomness_c: Output<u64>, kernel: Kernel<u64>) {
        match self {
            Conv2d {
                data,
                sfhe,
                masks,
                noise,
                result: _,
            } => {
                let mut tmp_images = Vec::new();
                let mut kernel_vec: Vec<_> = vec![std::ptr::null(); data.out_chans as usize];
                let mut server_randomness_vec: Vec<_> =
                    vec![std::ptr::null(); data.out_chans as usize];

                for out_c in 0..data.out_chans as usize {
                    server_randomness_vec[out_c] = server_randomness_c
                        .slice(s![0, out_c, .., ..])
                        .as_slice()
                        .expect("Error converting server randomness")
                        .as_ptr();

                    // No easy way to convert directly to double pointer so create a vector for
                    // each double pointer, get a pointer to it, and push it to tmp_images
                    // so that it doesn't get dropped.
                    //
                    // At the end of the outer scope, tmp_images will be dropped after
                    // kernel_vec, so we won't have a use after free kind of situation.
                    let mut tmp_image: Vec<*const u64> =
                        vec![std::ptr::null(); data.inp_chans as usize];
                    for (inp_c, tmp_i) in tmp_image.iter_mut().enumerate() {
                        *tmp_i = kernel
                            .slice(s![out_c, inp_c, .., ..])
                            .into_slice()
                            .expect("Error converting kernel")
                            .as_ptr();
                    }
                    kernel_vec[out_c] = tmp_image.as_ptr();
                    // This ensures that tmp_image lives on past the scope of the loop.
                    tmp_images.push(tmp_image);
                }
                // The Rust simply passes these pointers between various C++ functions and
                // doesn't need to reason about the types at all. The pointer
                // types for Conv/FC are different, so to avoid a lot of
                // unneeded verbosity simply have rust ignore pointer types via transmute
                unsafe {
                    *masks = server_conv_preprocess(kernel_vec.as_ptr(), *sfhe, data, 2);
                    *noise = conv_preprocess_noise(*sfhe, data, server_randomness_vec.as_ptr());
                }
            },
            FullyConnected {
                data,
                sfhe,
                masks,
                noise,
                result: _,
            } => {
                let mut kernel_vec: Vec<*const u64> =
                    vec![std::ptr::null(); data.filter_h as usize];
                let server_randomness_vec: *const u64;

                server_randomness_vec = server_randomness_c
                    .slice(s![0, .., .., ..])
                    .as_slice()
                    .expect("Error converting server randomness")
                    .as_ptr();

                for row in 0..data.filter_h as usize {
                    kernel_vec[row] = kernel
                        .slice(s![row, .., .., ..])
                        .into_slice()
                        .expect("Error converting kernel")
                        .as_ptr();
                }
                unsafe {
                    *masks = server_fc_preprocess(kernel_vec.as_ptr(), *sfhe, data);
                    *noise = fc_preprocess_noise(*sfhe, data, server_randomness_vec);
                }
            },
        };
    }

    pub fn process(&mut self, mut client_share: Vec<c_char>) -> Vec<std::os::raw::c_char> {
        match self {
            Conv2d {
                data,
                sfhe,
                masks,
                noise,
                result,
            } => {
                let mut enc_result_size: u64 = 0;
                unsafe {
                    *result = server_conv_online(
                        client_share.as_mut_ptr(),
                        *masks,
                        *sfhe,
                        data,
                        2,
                        *noise,
                        &mut enc_result_size,
                    );
                    std::slice::from_raw_parts(*result, enc_result_size as usize).to_vec()
                }
            },
            FullyConnected {
                data,
                sfhe,
                masks,
                noise,
                result,
            } => {
                let mut enc_result_size: u64 = 0;
                unsafe {
                    *result = server_fc_online(
                        client_share.as_mut_ptr(),
                        *masks,
                        *sfhe,
                        data,
                        *noise,
                        &mut enc_result_size,
                    );
                    std::slice::from_raw_parts(*result, enc_result_size as usize).to_vec()
                }
            },
        }
    }
}

impl<'a> Drop for SEALServerLinear<'a> {
    fn drop(&mut self) {
        match self {
            Conv2d {
                data,
                sfhe: _,
                masks,
                noise,
                result,
            } => unsafe { server_conv_free(data, *masks, *noise, *result, 2) },
            FullyConnected {
                data,
                sfhe: _,
                masks,
                noise,
                result,
            } => unsafe { server_fc_free(data, *masks, *noise, *result) },
        }
    }
}
