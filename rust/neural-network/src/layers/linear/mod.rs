use crate::{
    tensors::{Input, Kernel, Output},
    EvalMethod, Evaluate,
};
use algebra::{fixed_point::*, fp_64::Fp64Parameters, FpParameters, PrimeField};
use crypto_primitives::AdditiveShare;
use num_traits::{One, Zero};
use std::{
    marker::PhantomData,
    ops::{AddAssign, Mul},
};
use tch::nn::Module;

use crate::layers::LayerDims;
use LinearLayer::*;

pub mod convolution;
use convolution::*;

pub mod average_pooling;
use average_pooling::*;

pub mod fully_connected;
use fully_connected::*;

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub enum LinearLayer<F, C> {
    Conv2d {
        dims: LayerDims,
        params: Conv2dParams<F, C>,
    },
    FullyConnected {
        dims: LayerDims,
        params: FullyConnectedParams<F, C>,
    },
    AvgPool {
        dims: LayerDims,
        params: AvgPoolParams<F, C>,
    },
    Identity {
        dims: LayerDims,
    },
}

#[derive(Debug, Clone)]
pub enum LinearLayerInfo<F, C> {
    Conv2d {
        kernel: (usize, usize, usize, usize),
        padding: Padding,
        stride: usize,
    },
    FullyConnected,
    AvgPool {
        pool_h: usize,
        pool_w: usize,
        stride: usize,
        normalizer: C,
        _variable: PhantomData<F>,
    },
    Identity,
}

impl<F, C> LinearLayer<F, C> {
    pub fn dimensions(&self) -> LayerDims {
        match self {
            Conv2d { dims, .. }
            | FullyConnected { dims, .. }
            | AvgPool { dims, .. }
            | Identity { dims, .. } => *dims,
        }
    }

    pub fn input_dimensions(&self) -> (usize, usize, usize, usize) {
        self.dimensions().input_dimensions()
    }

    pub fn output_dimensions(&self) -> (usize, usize, usize, usize) {
        self.dimensions().output_dimensions()
    }

    pub fn all_dimensions(
        &self,
    ) -> (
        (usize, usize, usize, usize),
        (usize, usize, usize, usize),
        (usize, usize, usize, usize),
    ) {
        let input_dims = self.input_dimensions();
        let output_dims = self.output_dimensions();
        let kernel_dims = match self {
            Conv2d { dims: _, params: p } => p.kernel.dim(),
            FullyConnected { dims: _, params: p } => p.weights.dim(),
            _ => panic!("Identity/AvgPool layers do not have a kernel"),
        };
        (input_dims, output_dims, kernel_dims)
    }

    pub fn kernel_to_repr<P>(&self) -> Kernel<u64>
    where
        C: Copy + Into<FixedPoint<P>>,
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
    {
        match self {
            Conv2d { dims: _, params: p } => p.kernel.to_repr(),
            FullyConnected { dims: _, params: p } => p.weights.to_repr(),
            _ => panic!("Identity/AvgPool layers do not have a kernel"),
        }
    }

    pub fn eval_method(&self) -> crate::EvalMethod {
        match self {
            Conv2d { dims: _, params } => params.eval_method,
            FullyConnected { dims: _, params } => params.eval_method,
            _ => crate::EvalMethod::Naive,
        }
    }

    fn evaluate_naive(&self, input: &Input<F>, output: &mut Output<F>)
    where
        F: Zero + Mul<C, Output = F> + AddAssign + Copy,
        C: Copy + Into<F> + One + std::fmt::Debug,
    {
        match self {
            Conv2d { dims: _, params: p } => {
                p.conv2d_naive(input, output);
            }
            AvgPool { dims: _, params: p } => p.avg_pool_naive(input, output),
            FullyConnected { dims: _, params: p } => p.fully_connected_naive(input, output),
            Identity { dims: _ } => {
                *output = input.clone();
                let one = C::one();
                for elem in output.iter_mut() {
                    *elem = *elem * one;
                }
            }
        }
    }

    pub fn batch_evaluate(&self, input: &[Input<F>], output: &mut [Output<F>])
    where
        Self: Evaluate<F>,
    {
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = self.evaluate(inp);
        }
    }
}

impl<F, C> Evaluate<F> for LinearLayer<F, C>
where
    F: Zero + Mul<C, Output = F> + AddAssign + Copy,
    C: Copy + Into<F> + One + std::fmt::Debug,
{
    fn evaluate(&self, input: &Input<F>) -> Output<F> {
        self.evaluate_with_method(self.eval_method(), input)
    }

    default fn evaluate_with_method(&self, method: EvalMethod, input: &Input<F>) -> Output<F> {
        let mut output = Output::zeros(self.output_dimensions());
        match method {
            EvalMethod::Naive => self.evaluate_naive(input, &mut output),
            EvalMethod::TorchDevice(_) => {
                unimplemented!("cannot evaluate general networks with torch")
            }
        }
        output
    }
}

impl<P: FixedPointParameters> Evaluate<FixedPoint<P>>
    for LinearLayer<FixedPoint<P>, FixedPoint<P>>
{
    default fn evaluate_with_method(
        &self,
        method: EvalMethod,
        input: &Input<FixedPoint<P>>,
    ) -> Output<FixedPoint<P>> {
        let mut output = Output::zeros(self.output_dimensions());
        match method {
            EvalMethod::Naive => self.evaluate_naive(input, &mut output),
            EvalMethod::TorchDevice(_) => {
                unimplemented!("cannot evaluate general networks with torch")
            }
        }
        for elem in output.iter_mut() {
            elem.signed_reduce_in_place();
        }
        output
    }
}

impl<P> Evaluate<AdditiveShare<FixedPoint<P>>>
    for LinearLayer<AdditiveShare<FixedPoint<P>>, FixedPoint<P>>
where
    P: FixedPointParameters,
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    fn evaluate_with_method(
        &self,
        method: EvalMethod,
        input: &Input<AdditiveShare<FixedPoint<P>>>,
    ) -> Output<AdditiveShare<FixedPoint<P>>> {
        let mut output = Output::zeros(self.output_dimensions());
        match method {
            EvalMethod::Naive => self.evaluate_naive(input, &mut output),
            EvalMethod::TorchDevice(m) => {
                match self {
                    Conv2d { dims: _, params: p } => {
                        let conv2d = &p.tch_config;
                        // Send the tensor to the appropriate PyTorch device
                        let input_tensor = input.to_tensor().to_device(m);
                        output = conv2d
                            .as_ref()
                            .and_then(|cfg| Output::from_tensor(cfg.forward(&input_tensor)))
                            .expect("shape should be correct");
                    }
                    // FullyConnected { dims: _, params: p } => {
                    //     let fc = &p.tch_config;
                    //     let input_tensor = input.to_tensor();
                    //     // Send the tensor to the appropriate PyTorch device
                    //     input_tensor.to_device(m);
                    //     output = fc.as_ref().and_then(|cfg|
                    // Output::from_tensor(cfg.forward(&input_tensor))).expect("shape should be
                    // correct"); },
                    _ => self.evaluate_naive(input, &mut output),
                }
            }
        }
        output
    }
}

impl<P> Evaluate<FixedPoint<P>> for LinearLayer<FixedPoint<P>, FixedPoint<P>>
where
    P: FixedPointParameters,
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    fn evaluate_with_method(
        &self,
        method: EvalMethod,
        input: &Input<FixedPoint<P>>,
    ) -> Output<FixedPoint<P>> {
        let mut output = Output::zeros(self.output_dimensions());
        match method {
            EvalMethod::Naive => self.evaluate_naive(input, &mut output),
            EvalMethod::TorchDevice(m) => {
                match self {
                    Conv2d { dims: _, params: p } => {
                        let conv2d = &p.tch_config;
                        // Send the input to the appropriate PyTorch device
                        let input_tensor = input.to_tensor().to_device(m);
                        output = conv2d
                            .as_ref()
                            .and_then(|cfg| Output::from_tensor(cfg.forward(&input_tensor)))
                            .expect("shape should be correct");
                    }
                    // FullyConnected { dims: _, params: p } => {
                    //     let fc = &p.tch_config;
                    //     // Send the input to the appropriate PyTorch device
                    //     let input_tensor = input.to_tensor().to_device(m);
                    //     output = fc.as_ref().and_then(|cfg|
                    // Output::from_tensor(cfg.forward(&input_tensor))).expect("shape should be
                    // correct"); },
                    _ => self.evaluate_naive(input, &mut output),
                }
            }
        }
        output
    }
}

impl<F, C> LinearLayerInfo<F, C> {
    pub fn evaluate_naive(&self, input: &Input<F>, output: &mut Output<F>)
    where
        F: Zero + Mul<C, Output = F> + AddAssign + Copy,
        C: Copy + One + std::fmt::Debug,
    {
        match self {
            LinearLayerInfo::AvgPool {
                pool_h,
                pool_w,
                stride,
                normalizer,
                ..
            } => {
                let params = AvgPoolParams::new(*pool_h, *pool_w, *stride, *normalizer);
                params.avg_pool_naive(input, output)
            }
            LinearLayerInfo::Identity => {
                *output = input.clone();
                let one = C::one();
                for elem in output.iter_mut() {
                    *elem = *elem * one;
                }
            }
            _ => unreachable!(),
        }
    }
}

impl<'a, F, C: Clone> From<&'a LinearLayer<F, C>> for LinearLayerInfo<F, C> {
    fn from(other: &'a LinearLayer<F, C>) -> Self {
        match other {
            LinearLayer::Conv2d { params, .. } => LinearLayerInfo::Conv2d {
                kernel: params.kernel.dim(),
                padding: params.padding,
                stride: params.stride,
            },
            LinearLayer::FullyConnected { .. } => LinearLayerInfo::FullyConnected,
            LinearLayer::AvgPool { params, .. } => LinearLayerInfo::AvgPool {
                pool_h: params.pool_h,
                pool_w: params.pool_w,
                stride: params.stride,
                normalizer: params.normalizer.clone(),
                _variable: std::marker::PhantomData,
            },
            LinearLayer::Identity { .. } => LinearLayerInfo::Identity,
        }
    }
}
