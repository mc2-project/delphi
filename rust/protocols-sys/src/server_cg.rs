use crate::*;
use neural_network::{
    layers::{convolution::Padding, LinearLayer},
    tensors::{Kernel, Output},
};
use std::os::raw::c_char;

pub struct Conv2D<'a> {
    data: Metadata,
    sfhe: &'a ServerFHE,
    masks: *mut *mut *mut *mut c_char,
    shares: Option<ServerShares>,
}

pub struct FullyConnected<'a> {
    data: Metadata,
    sfhe: &'a ServerFHE,
    masks: *mut *mut c_char,
    shares: Option<ServerShares>,
}

pub enum SealServerCG<'a> {
    Conv2D(Conv2D<'a>),
    FullyConnected(FullyConnected<'a>),
}

pub trait ServerCG {
    type Keys;

    fn new<F, C>(sfhe: Self::Keys, layer: &LinearLayer<F, C>, kernel: &Kernel<u64>) -> Self;

    fn preprocess(&mut self, linear_share: &Output<u64>);

    fn process(&mut self, client_share: Vec<c_char>) -> Vec<c_char>;
}

impl<'a> SealServerCG<'a> {
    pub fn preprocess(&mut self, linear_share: &Output<u64>) {
        match self {
            Self::Conv2D(s) => s.preprocess(linear_share),
            Self::FullyConnected(s) => s.preprocess(linear_share),
        }
    }

    pub fn process(&mut self, client_share: Vec<c_char>) -> Vec<c_char> {
        match self {
            Self::Conv2D(s) => s.process(client_share),
            Self::FullyConnected(s) => s.process(client_share),
        }
    }
}

impl<'a> ServerCG for Conv2D<'a> {
    type Keys = &'a ServerFHE;

    fn new<F, C>(sfhe: &'a ServerFHE, layer: &LinearLayer<F, C>, kernel: &Kernel<u64>) -> Self {
        let (input_dims, _, kernel_dims) = layer.all_dimensions();
        let params = match layer {
            LinearLayer::Conv2d { params, .. } => params,
            _ => panic!("Incorrect Layer"),
        };
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
            )
        };
        let mut tmp_images = Vec::new();
        let mut kernel_vec: Vec<_> = vec![std::ptr::null(); data.out_chans as usize];

        for out_c in 0..data.out_chans as usize {
            // No easy way to convert directly to double pointer so create a vector for
            // each double pointer, get a pointer to it, and push it to tmp_images
            // so that it doesn't get dropped.
            //
            // At the end of the outer scope, tmp_images will be dropped after
            // kernel_vec, so we won't have a use after free kind of situation.
            let mut tmp_image: Vec<*const u64> = vec![std::ptr::null(); data.inp_chans as usize];
            for (inp_c, tmp_i) in tmp_image.iter_mut().enumerate() {
                *tmp_i = kernel
                    .slice(s![out_c, inp_c, .., ..])
                    .to_slice()
                    .expect("Error converting kernel")
                    .as_ptr();
            }
            kernel_vec[out_c] = tmp_image.as_ptr();
            // This ensures that tmp_image lives on past the scope of the loop.
            tmp_images.push(tmp_image);
        }
        let masks = unsafe { server_conv_preprocess(sfhe, &data, kernel_vec.as_ptr()) };
        Self {
            data,
            sfhe,
            masks,
            shares: None,
        }
    }

    fn preprocess(&mut self, linear_share: &Output<u64>) {
        let mut linear_vec: Vec<_> = vec![std::ptr::null(); self.data.out_chans as usize];

        for out_c in 0..self.data.out_chans as usize {
            linear_vec[out_c] = linear_share
                .slice(s![0, out_c, .., ..])
                .as_slice()
                .expect("Error converting server randomness")
                .as_ptr();
        }
        self.shares = Some(unsafe {
            server_conv_preprocess_shares(self.sfhe, &self.data, linear_vec.as_ptr())
        });
    }

    fn process(&mut self, mut client_share: Vec<c_char>) -> Vec<c_char> {
        let mut shares = self.shares.unwrap();
        let client_share_ct = SerialCT {
            inner: client_share.as_mut_ptr(),
            size: client_share.len() as u64,
        };
        unsafe {
            server_conv_online(
                self.sfhe,
                &self.data,
                client_share_ct,
                self.masks,
                &mut shares,
            )
        };
        self.shares = Some(shares);
        // Return ciphertexts as vectors
        let linear_ct_vec = unsafe {
            std::slice::from_raw_parts(shares.linear_ct.inner, shares.linear_ct.size as usize)
                .to_vec()
        };
        linear_ct_vec
    }
}

impl<'a> ServerCG for FullyConnected<'a> {
    type Keys = &'a ServerFHE;

    fn new<F, C>(sfhe: &'a ServerFHE, layer: &LinearLayer<F, C>, kernel: &Kernel<u64>) -> Self {
        let (input_dims, output_dims, _) = layer.all_dimensions();
        let data = unsafe {
            fc_metadata(
                sfhe.encoder,
                (input_dims.1 * input_dims.2 * input_dims.3) as i32,
                output_dims.1 as i32,
            )
        };

        let mut kernel_vec: Vec<*const u64> = vec![std::ptr::null(); data.filter_h as usize];

        for row in 0..data.filter_h as usize {
            kernel_vec[row] = kernel
                .slice(s![row, .., .., ..])
                .to_slice()
                .expect("Error converting kernel")
                .as_ptr();
        }
        let masks = unsafe { server_fc_preprocess(sfhe, &data, kernel_vec.as_ptr()) };
        Self {
            data,
            sfhe,
            masks,
            shares: None,
        }
    }

    fn preprocess(&mut self, linear_share: &Output<u64>) {
        let linear: *const u64;

        linear = linear_share
            .slice(s![0, .., .., ..])
            .as_slice()
            .expect("Error converting server randomness")
            .as_ptr();

        self.shares = Some(unsafe { server_fc_preprocess_shares(self.sfhe, &self.data, linear) });
    }

    fn process(&mut self, mut client_share: Vec<c_char>) -> Vec<c_char> {
        let mut shares = self.shares.unwrap();
        let client_share_ct = SerialCT {
            inner: client_share.as_mut_ptr(),
            size: client_share.len() as u64,
        };
        unsafe {
            server_fc_online(
                self.sfhe,
                &self.data,
                client_share_ct,
                self.masks,
                &mut shares,
            )
        };
        self.shares = Some(shares);
        // Return ciphertexts as vectors
        let linear_ct_vec = unsafe {
            std::slice::from_raw_parts(shares.linear_ct.inner, shares.linear_ct.size as usize)
                .to_vec()
        };
        linear_ct_vec
    }
}

impl<'a> Drop for Conv2D<'a> {
    fn drop(&mut self) {
        unsafe { server_conv_free(&self.data, self.masks, &mut self.shares.unwrap()) };
    }
}

impl<'a> Drop for FullyConnected<'a> {
    fn drop(&mut self) {
        unsafe { server_fc_free(&self.data, self.masks, &mut self.shares.unwrap()) };
    }
}
