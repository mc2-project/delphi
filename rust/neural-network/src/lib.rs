#![allow(incomplete_features)]
#![feature(specialization)]
use crate::tensors::{Input, Output};
use ndarray::ArrayView;
use num_traits::{One, Zero};
use std::{
    io::Read,
    ops::{AddAssign, Mul, MulAssign},
};

#[macro_use]
pub extern crate ndarray;

pub extern crate npy;
extern crate npy_derive;

pub mod layers;
pub mod tensors;

use layers::{
    Layer, LayerInfo,
    LinearLayer::{Conv2d, FullyConnected},
};
use npy::NpyData;

pub trait Evaluate<F> {
    #[inline]
    fn evaluate(&self, input: &Input<F>) -> Output<F> {
        self.evaluate_with_method(EvalMethod::default(), input)
    }

    fn evaluate_with_method(&self, method: EvalMethod, input: &Input<F>) -> Output<F>;
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum EvalMethod {
    Naive,
    TorchDevice(tch::Device),
}

impl Default for EvalMethod {
    fn default() -> Self {
        EvalMethod::Naive
    }
}

impl EvalMethod {
    #[inline]
    pub fn with_tch_device(dev: tch::Device) -> Self {
        EvalMethod::TorchDevice(dev)
    }

    #[inline]
    pub fn with_var_store(vs: &tch::nn::Path) -> Self {
        EvalMethod::TorchDevice(vs.device())
    }
}

/// `NeuralNetwork` represents a neural network as a sequence of `Layer`s that
/// are to be evaluated sequentially.
// TODO: add residual layers.
// Probably by generalizing to a DAG instead of a Vec.
#[derive(Debug, Default)]
pub struct NeuralNetwork<F, C = F> {
    pub eval_method: EvalMethod,
    pub layers: Vec<Layer<F, C>>,
}

/// Describes the architecture and topology of the network
#[derive(Clone, Debug, Default)]
pub struct NeuralArchitecture<F, C = F> {
    pub layers: Vec<LayerInfo<F, C>>,
}

impl<F, C> NeuralNetwork<F, C>
where
    C: std::convert::From<f64>,
{
    /// `validate` ensures that the layers of `self` from a valid sequence.
    /// That is, `validate` checks that the output dimensions of the i-th layer
    /// are equal to the input dimensions of the (i+1)-th layer.
    // TODO: make this work with residual layers.
    // When we switch to DAGs, check that for every layer, the parent layer(s)
    // have matching output dimensions.
    pub fn validate(&self) -> bool {
        let len = self.layers.len();
        if len == 0 || len == 1 {
            true
        } else {
            let mut result = true;
            for (i, (layer, next_layer)) in self.layers.iter().zip(&self.layers[1..]).enumerate() {
                if layer.output_dimensions() != next_layer.input_dimensions() {
                    eprintln!(
                        "layer {} is incorrect: expected {:?}, got {:?}",
                        i,
                        layer.output_dimensions(),
                        next_layer.input_dimensions(),
                    );
                }
                result &= layer.output_dimensions() == next_layer.input_dimensions()
            }
            result
        }
    }

    pub fn from_numpy(&mut self, weights_path: &str) -> Result<(), ndarray::ShapeError> {
        // Deserialize numpy weights into a 1-d vector
        let mut buf = vec![];
        std::fs::File::open(weights_path)
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        let weights: Vec<f64> = NpyData::from_bytes(&buf).unwrap().to_vec();
        let mut weights_idx = 0;

        // npy can't do multi-dimensional numpy serialization so all the weights are
        // stored as a single flattened array.
        for layer in self.layers.iter_mut() {
            let (kernel, bias) = match layer {
                Layer::LL(Conv2d { dims: _, params }) => (&params.kernel, &params.bias),
                Layer::LL(FullyConnected { dims: _, params }) => (&params.weights, &params.bias),
                _ => continue,
            };
            let kernel_dims = kernel.dim();
            let kernel_size = kernel_dims.0 * kernel_dims.1 * kernel_dims.2 * kernel_dims.3;
            let new_kernel = ArrayView::from_shape(
                kernel_dims,
                &weights[weights_idx..(weights_idx + kernel_size)],
            )?;
            weights_idx += kernel_size;

            let bias_dims = bias.dim();
            let bias_size = bias_dims.0;
            let new_bias =
                ArrayView::from_shape(bias_dims, &weights[weights_idx..(weights_idx + bias_size)])?;
            weights_idx += bias_size;

            match layer {
                Layer::LL(Conv2d { dims: _, params }) => params.kernel.iter_mut(),
                Layer::LL(FullyConnected { dims: _, params }) => params.weights.iter_mut(),
                _ => unreachable!(),
            }
            .zip(new_kernel.iter())
            .for_each(|(o, n)| *o = (*n).into());

            match layer {
                Layer::LL(Conv2d { dims: _, params }) => params.bias.iter_mut(),
                Layer::LL(FullyConnected { dims: _, params }) => params.bias.iter_mut(),
                _ => unreachable!(),
            }
            .zip(new_bias.iter())
            .for_each(|(o, n)| *o = (*n).into());
        }
        Ok(())
    }
}

impl<'a, F, C: Clone> From<&'a NeuralNetwork<F, C>> for NeuralArchitecture<F, C> {
    fn from(other: &'a NeuralNetwork<F, C>) -> Self {
        let layers = other.layers.iter().map(|layer| layer.into()).collect();
        Self { layers }
    }
}

impl<F> Evaluate<F> for NeuralNetwork<F, F>
where
    F: One
        + Zero
        + Mul<Output = F>
        + AddAssign
        + MulAssign
        + PartialOrd
        + Copy
        + From<f64>
        + std::fmt::Debug,
{
    #[inline]
    fn evaluate(&self, input: &Input<F>) -> Output<F> {
        self.evaluate_with_method(self.eval_method, input)
    }

    /// `evaluate` takes an `input`, evaluates `self` over `input`, and returns
    /// the result. Panics if `self.validate() == false`.
    // TODO: make this work with residual layers.
    // When we switch to DAGs, check that for every layer, the parent layer(s)
    // have matching output dimensions.
    default fn evaluate_with_method(&self, method: EvalMethod, input: &Input<F>) -> Output<F> {
        assert!(self.validate());
        if self.layers.len() == 0 {
            input.clone()
        } else {
            let mut input = input.clone();
            for layer in &self.layers {
                let output = layer.evaluate_with_method(method, &input);
                input = output;
            }
            input
        }
    }
}
