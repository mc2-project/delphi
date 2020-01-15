use crate::{
    tensors::{Input, Output},
    Evaluate,
};
use num_traits::{One, Zero};
use std::ops::{AddAssign, Mul, MulAssign};

mod linear;
mod non_linear;

pub use linear::*;
pub use non_linear::*;
use Layer::*;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct LayerDims {
    /// Dimension of the input to a layer: `(batch_size, channels, height,
    /// width)`
    pub input_dims:  (usize, usize, usize, usize),
    /// Dimension of the output of a layer: `(batch_size, channels, height,
    /// width)`
    pub output_dims: (usize, usize, usize, usize),
}

impl LayerDims {
    pub fn input_dimensions(&self) -> (usize, usize, usize, usize) {
        self.input_dims
    }

    pub fn output_dimensions(&self) -> (usize, usize, usize, usize) {
        self.output_dims
    }
}

#[derive(Debug)]
pub enum Layer<F, C> {
    LL(LinearLayer<F, C>),
    NLL(NonLinearLayer<F, C>),
}

impl<F, C> Layer<F, C> {
    #[inline]
    pub fn is_linear(&self) -> bool {
        match self {
            LL(_) => true,
            NLL(_) => false,
        }
    }

    #[inline]
    pub fn is_non_linear(&self) -> bool {
        !self.is_linear()
    }

    pub fn input_dimensions(&self) -> (usize, usize, usize, usize) {
        match self {
            LL(l) => l.input_dimensions(),
            NLL(l) => l.input_dimensions(),
        }
    }

    pub fn output_dimensions(&self) -> (usize, usize, usize, usize) {
        match self {
            LL(l) => l.output_dimensions(),
            NLL(l) => l.output_dimensions(),
        }
    }
}
impl<F, C> Evaluate<F> for Layer<F, C>
where
    F: Zero + One + MulAssign + Mul<C, Output = F> + AddAssign + PartialOrd<C> + Copy,
    C: std::fmt::Debug + Copy + Into<F> + From<f64> + Zero + One,
{
    fn evaluate(&self, input: &Input<F>) -> Output<F> {
        match self {
            LL(l) => l.evaluate(input),
            NLL(l) => l.evaluate(input),
        }
    }

    fn evaluate_with_method(&self, method: crate::EvalMethod, input: &Input<F>) -> Output<F> {
        match self {
            LL(l) => l.evaluate_with_method(method, input),
            NLL(l) => l.evaluate_with_method(method, input),
        }
    }
}

#[derive(Debug, Clone)]
pub enum LayerInfo<F, C> {
    LL(LayerDims, LinearLayerInfo<F, C>),
    NLL(LayerDims, NonLinearLayerInfo<F, C>),
}

impl<F, C> LayerInfo<F, C> {
    #[inline]
    pub fn is_linear(&self) -> bool {
        match self {
            LayerInfo::LL(..) => true,
            LayerInfo::NLL(..) => false,
        }
    }

    #[inline]
    pub fn is_non_linear(&self) -> bool {
        !self.is_linear()
    }

    pub fn input_dimensions(&self) -> (usize, usize, usize, usize) {
        match self {
            LayerInfo::LL(l, _) => l.input_dimensions(),
            LayerInfo::NLL(l, _) => l.input_dimensions(),
        }
    }

    pub fn output_dimensions(&self) -> (usize, usize, usize, usize) {
        match self {
            LayerInfo::LL(l, _) => l.output_dimensions(),
            LayerInfo::NLL(l, _) => l.output_dimensions(),
        }
    }
}

impl<'a, F, C: Clone> From<&'a Layer<F, C>> for LayerInfo<F, C> {
    fn from(other: &'a Layer<F, C>) -> Self {
        match other {
            LL(LinearLayer::Conv2d { dims, params }) => LayerInfo::LL(
                *dims,
                LinearLayerInfo::Conv2d {
                    kernel:  params.kernel.dim(),
                    padding: params.padding,
                    stride:  params.stride,
                },
            ),
            // TODO: Is there a way to match all of these with one statement
            LL(LinearLayer::FullyConnected { dims, params: _ }) => {
                LayerInfo::LL(*dims, LinearLayerInfo::FullyConnected)
            },
            LL(LinearLayer::AvgPool { dims, params }) => LayerInfo::LL(
                *dims,
                LinearLayerInfo::AvgPool {
                    pool_h:     params.pool_h,
                    pool_w:     params.pool_w,
                    stride:     params.stride,
                    normalizer: params.normalizer.clone(),
                    _variable:  std::marker::PhantomData,
                },
            ),
            LL(LinearLayer::Identity { dims }) => {
                LayerInfo::LL(*dims, LinearLayerInfo::FullyConnected)
            },
            NLL(NonLinearLayer::ReLU(dims)) => LayerInfo::NLL(*dims, NonLinearLayerInfo::ReLU),
            NLL(NonLinearLayer::PolyApprox { dims, poly, .. }) => LayerInfo::NLL(
                *dims,
                NonLinearLayerInfo::PolyApprox {
                    poly: poly.clone(),
                    _v:   std::marker::PhantomData,
                },
            ),
        }
    }
}
