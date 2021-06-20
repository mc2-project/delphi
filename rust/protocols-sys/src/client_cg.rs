use crate::*;
use algebra::{fixed_point::*, fp_64::Fp64Parameters, FpParameters, PrimeField};
use crypto_primitives::additive_share::AdditiveShare;
use neural_network::{
    layers::{convolution::Padding, LinearLayerInfo},
    tensors::{Input, Output},
};
use std::os::raw::c_char;

pub struct Conv2D<'a> {
    data: Metadata,
    cfhe: &'a ClientFHE,
    shares: Option<ClientShares>,
}

pub struct FullyConnected<'a> {
    data: Metadata,
    cfhe: &'a ClientFHE,
    shares: Option<ClientShares>,
}

pub enum SealClientCG<'a> {
    Conv2D(Conv2D<'a>),
    FullyConnected(FullyConnected<'a>),
}

pub trait ClientCG {
    type Keys;

    fn new<F, C>(
        cfhe: Self::Keys,
        layer_info: &LinearLayerInfo<F, C>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
    ) -> Self
    where
        Self: std::marker::Sized;

    fn preprocess(&mut self, r: &Input<u64>) -> Vec<c_char>;

    fn decrypt(&mut self, linear_ct: Vec<c_char>);

    fn postprocess<P>(&self, linear_share: &mut Output<AdditiveShare<FixedPoint<P>>>)
    where
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>;
}

impl<'a> SealClientCG<'a> {
    pub fn preprocess(&mut self, r: &Input<u64>) -> Vec<c_char> {
        match self {
            Self::Conv2D(s) => s.preprocess(r),
            Self::FullyConnected(s) => s.preprocess(r),
        }
    }

    pub fn decrypt(&mut self, linear_ct: Vec<c_char>) {
        match self {
            Self::Conv2D(s) => s.decrypt(linear_ct),
            Self::FullyConnected(s) => s.decrypt(linear_ct),
        };
    }

    pub fn postprocess<P>(&self, linear_share: &mut Output<AdditiveShare<FixedPoint<P>>>)
    where
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
    {
        match self {
            Self::Conv2D(s) => ClientCG::postprocess::<P>(s, linear_share),
            Self::FullyConnected(s) => ClientCG::postprocess::<P>(s, linear_share),
        };
    }
}

impl<'a> ClientCG for Conv2D<'a> {
    type Keys = &'a ClientFHE;

    fn new<F, C>(
        cfhe: &'a ClientFHE,
        layer_info: &LinearLayerInfo<F, C>,
        input_dims: (usize, usize, usize, usize),
        _output_dims: (usize, usize, usize, usize),
    ) -> Self {
        let (kernel, padding, stride) = match layer_info {
            LinearLayerInfo::Conv2d {
                kernel,
                padding,
                stride,
            } => (kernel, padding, stride),
            _ => panic!("Incorrect Layer Type"),
        };
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
            )
        };
        Self {
            data,
            cfhe,
            shares: None,
        }
    }

    fn preprocess(&mut self, r: &Input<u64>) -> Vec<c_char> {
        // Convert client secret share to raw pointers for C FFI
        let r_c: Vec<*const u64> = (0..self.data.inp_chans)
            .into_iter()
            .map(|inp_c| {
                r.slice(s![0, inp_c, .., ..])
                    .as_slice()
                    .expect("Error converting client share")
                    .as_ptr()
            })
            .collect();
        let shares = unsafe { client_conv_preprocess(self.cfhe, &self.data, r_c.as_ptr()) };
        let ct_vec = unsafe {
            std::slice::from_raw_parts(shares.input_ct.inner, shares.input_ct.size as usize)
                .to_vec()
        };
        self.shares = Some(shares);
        ct_vec
    }

    fn decrypt(&mut self, mut linear_ct: Vec<c_char>) {
        let mut shares = self.shares.unwrap();
        // Copy the received ciphertexts into share struct
        shares.linear_ct = SerialCT {
            inner: linear_ct.as_mut_ptr(),
            size: linear_ct.len() as u64,
        };
        // Decrypt everything
        unsafe { client_conv_decrypt(self.cfhe, &self.data, &mut shares) };
        self.shares = Some(shares);
    }

    fn postprocess<P>(&self, linear_share: &mut Output<AdditiveShare<FixedPoint<P>>>)
    where
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
    {
        let shares = self.shares.unwrap();
        for chan in 0..self.data.out_chans as usize {
            for row in 0..self.data.output_h as usize {
                for col in 0..self.data.output_w as usize {
                    let idx = (row * (self.data.output_w as usize) + col) as isize;
                    let linear_val =
                        unsafe { *(*(shares.linear.offset(chan as isize))).offset(idx as isize) };
                    linear_share[[0, chan, row, col]] = AdditiveShare::new(
                        FixedPoint::with_num_muls(P::Field::from_repr(linear_val.into()), 1),
                    );
                }
            }
        }
    }
}

impl<'a> ClientCG for FullyConnected<'a> {
    type Keys = &'a ClientFHE;

    fn new<F, C>(
        cfhe: &'a ClientFHE,
        _layer_info: &LinearLayerInfo<F, C>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
    ) -> Self {
        let data = unsafe {
            fc_metadata(
                cfhe.encoder,
                (input_dims.1 * input_dims.2 * input_dims.3) as i32,
                output_dims.1 as i32,
            )
        };
        Self {
            data,
            cfhe,
            shares: None,
        }
    }

    fn preprocess(&mut self, r: &Input<u64>) -> Vec<c_char> {
        // Convert client secret share to raw pointers for C FFI
        let r_c: *const u64 = r
            .slice(s![0, .., .., ..])
            .as_slice()
            .expect("Error converting client share")
            .as_ptr();
        let shares = unsafe { client_fc_preprocess(self.cfhe, &self.data, r_c) };
        let ct_vec = unsafe {
            std::slice::from_raw_parts(shares.input_ct.inner, shares.input_ct.size as usize)
                .to_vec()
        };
        self.shares = Some(shares);
        ct_vec
    }

    fn decrypt(&mut self, mut linear_ct: Vec<c_char>) {
        let mut shares = self.shares.unwrap();
        // Copy the received ciphertexts into share struct
        shares.linear_ct = SerialCT {
            inner: linear_ct.as_mut_ptr(),
            size: linear_ct.len() as u64,
        };
        // Decrypt everything
        unsafe { client_fc_decrypt(self.cfhe, &self.data, &mut shares) };
        self.shares = Some(shares);
    }

    fn postprocess<P>(&self, linear_share: &mut Output<AdditiveShare<FixedPoint<P>>>)
    where
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
    {
        let shares = self.shares.unwrap();
        for row in 0..self.data.filter_h as usize {
            let linear_val = unsafe { *(*(shares.linear.offset(0))).offset(row as isize) };
            linear_share[[0, row, 0, 0]] = AdditiveShare::new(FixedPoint::with_num_muls(
                P::Field::from_repr(linear_val.into()),
                1,
            ));
        }
    }
}

impl<'a> Drop for Conv2D<'a> {
    fn drop(&mut self) {
        unsafe { client_conv_free(&self.data, &mut self.shares.unwrap()) }
    }
}

impl<'a> Drop for FullyConnected<'a> {
    fn drop(&mut self) {
        unsafe { client_fc_free(&mut self.shares.unwrap()) };
    }
}
