use crate::tensors::{Input, Kernel, Output};
use algebra::{fp_64::Fp64Parameters, FixedPoint, FixedPointParameters, FpParameters, PrimeField};
use crypto_primitives::AdditiveShare;
use num_traits::Zero;
use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Mul},
};
use tch::nn;

#[derive(Debug)]
pub struct FullyConnectedParams<F, C> {
    pub weights: Kernel<C>,
    pub bias: Kernel<C>,
    pub tch_config: Option<nn::Linear>,
    pub eval_method: crate::EvalMethod,
    _variable: PhantomData<F>,
}

unsafe impl<F, C> Send for FullyConnectedParams<F, C> {}
unsafe impl<F, C> Sync for FullyConnectedParams<F, C> {}

impl<F, C> FullyConnectedParams<F, C>
where
    F: Zero + Mul<C, Output = F> + AddAssign + Add<Output = F> + Copy,
    C: Copy + Into<F>,
{
    pub fn new(weights: Kernel<C>, bias: Kernel<C>) -> Self {
        let kernel_dims = weights.dim();
        let bias_dims = bias.dim();
        assert!(
            (bias_dims.0 == kernel_dims.0)
                && (bias_dims.1 == 1)
                && (bias_dims.2 == 1)
                && (bias_dims.3 == 1)
        );
        Self {
            weights,
            bias,
            tch_config: None,
            eval_method: crate::EvalMethod::Naive,
            _variable: PhantomData,
        }
    }

    pub fn calculate_output_size(
        &self,
        (batch_size, in_channels, in_height, in_width): (usize, usize, usize, usize),
    ) -> (usize, usize, usize, usize) {
        let (num, w_channels, w_height, w_width) = self.weights.dim();
        assert_eq!(w_height, in_height);
        assert_eq!(w_width, in_width);
        assert_eq!(w_channels, in_channels);

        let out_height = 1;
        let out_width = 1;
        let out_channels = num;
        (batch_size, out_channels, out_height, out_width)
    }

    pub fn fully_connected_naive(&self, input: &Input<F>, out: &mut Output<F>) {
        let (batch_size, in_channels, in_height, in_width) = input.dim();
        let (num, w_channels, w_height, w_width) = self.weights.dim();
        let (o_batch_size, o_channels, ..) = out.dim();
        assert!(
            (o_batch_size == batch_size)
                & (w_height == in_height)
                & (w_width == in_width)
                & (in_channels == w_channels)
                & (o_channels == num),
            "Shape doesn't match: input: {:?}, weights: {:?}, output: {:?}",
            input.dim(),
            self.weights.dim(),
            out.dim()
        );

        let k = in_height * in_width;
        let in_slice = s![0..batch_size, 0..in_channels, 0..in_height, 0..in_width];
        let w_slice = s![0..num, 0..w_channels, 0..w_height, 0..w_width];
        let inp = input
            .slice(&in_slice)
            .into_shape((batch_size, in_channels, k))
            .expect("Reshape should work");
        let weights = self
            .weights
            .slice(&w_slice)
            .into_shape((num, w_channels, k))
            .expect("Reshape should work");

        let i_zero = ndarray::Axis(0);
        out.axis_iter_mut(i_zero)
            .zip(inp.axis_iter(i_zero))
            .for_each(|(mut out, inp)| {
                // for each output-input pair in batch
                weights
                    .axis_iter(i_zero)
                    .zip(out.axis_iter_mut(i_zero))
                    .for_each(|(weight, mut out)| {
                        weight
                            .axis_iter(i_zero)
                            .zip(inp.axis_iter(i_zero))
                            .for_each(|(weight, inp)| unsafe {
                                let elt = out.uget_mut((0, 0));
                                *elt = (0..k)
                                    .fold(*elt, move |s, x| s + *inp.uget(x) * *weight.uget(x));
                            })
                    });
            });
        // Add the appropiate bias to each channel of output
        out.outer_iter_mut().for_each(|mut batch| {
            batch
                .outer_iter_mut()
                .enumerate()
                .for_each(|(i, mut view)| {
                    let b = unsafe { *self.bias.uget((i, 0, 0, 0)) };
                    view.iter_mut().for_each(|e| *e += b.into());
                });
        });
    }
}

impl<P: FixedPointParameters> FullyConnectedParams<AdditiveShare<FixedPoint<P>>, FixedPoint<P>>
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    pub fn new_with_gpu(
        vs: &nn::Path,
        weights: Kernel<FixedPoint<P>>,
        bias: Kernel<FixedPoint<P>>,
    ) -> Self {
        let (out_channels, in_channels, ..) = weights.dim();
        let device = vs.device();
        let weights_tensor = weights.to_tensor().to_device(device);
        let bias_tensor = bias
            .to_tensor()
            .reshape(&[bias.dim().0 as i64])
            .to_device(device);
        let mut out = Self::new(weights, bias);
        out.eval_method = crate::EvalMethod::TorchDevice(device);

        let mut tch_config = nn::linear(
            vs,
            in_channels as i64,
            out_channels as i64,
            Default::default(),
        );
        tch_config.ws = weights_tensor;
        tch_config.bs = bias_tensor;
        out.tch_config = Some(tch_config);
        out
    }
}
