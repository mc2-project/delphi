use crate::{
    layers::LayerDims,
    tensors::{Input, Output},
};
use algebra::Polynomial;
use num_traits::{One, Zero};
use std::{
    marker::PhantomData,
    ops::{AddAssign, Mul, MulAssign},
};

use crate::Evaluate;
use NonLinearLayer::*;

#[derive(Debug, Clone)]
pub enum NonLinearLayer<F, C = F> {
    ReLU(LayerDims),
    PolyApprox {
        dims: LayerDims,
        poly: Polynomial<C>,
        _v:   PhantomData<F>,
    },
}

#[derive(Debug, Clone)]
pub enum NonLinearLayerInfo<F, C> {
    ReLU,
    PolyApprox {
        poly: Polynomial<C>,
        _v:   PhantomData<F>,
    },
}

impl<F, C> NonLinearLayer<F, C> {
    pub fn dimensions(&self) -> LayerDims {
        match self {
            ReLU(dims) | PolyApprox { dims, .. } => *dims,
        }
    }

    pub fn input_dimensions(&self) -> (usize, usize, usize, usize) {
        self.dimensions().input_dimensions()
    }

    pub fn output_dimensions(&self) -> (usize, usize, usize, usize) {
        self.dimensions().input_dimensions()
    }
}
impl<F, C> Evaluate<F> for NonLinearLayer<F, C>
where
    F: One + Zero + Mul<C, Output = F> + AddAssign + MulAssign + PartialOrd<C> + Copy,
    C: Copy + From<f64> + Zero,
{
    fn evaluate_with_method(&self, _: crate::EvalMethod, input: &Input<F>) -> Output<F> {
        assert_eq!(self.input_dimensions(), input.dim());
        let mut output = Output::zeros(self.output_dimensions());
        match self {
            ReLU(_) => {
                let zero = C::zero();
                let f_zero = F::zero();
                for (&inp, out) in input.iter().zip(&mut output) {
                    *out = if inp > zero { inp } else { f_zero };
                }
            },
            PolyApprox { dims: _d, poly, .. } => {
                for (&inp, out) in input.iter().zip(&mut output) {
                    *out = poly.evaluate(inp);
                }
            },
        };
        output
    }
}
