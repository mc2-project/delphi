use crate::tensors::{Input, Output};
use num_traits::Zero;
use std::{
    marker::PhantomData,
    ops::{AddAssign, Mul},
};

#[derive(Debug, Clone)]
pub struct AvgPoolParams<F, C> {
    pub pool_h:     usize,
    pub pool_w:     usize,
    pub stride:     usize,
    pub normalizer: C,
    _variable:      PhantomData<F>,
}

impl<F, C> AvgPoolParams<F, C>
where
    F: Zero + Mul<C, Output = F> + AddAssign + Copy,
    C: Copy,
{
    pub fn new(pool_h: usize, pool_w: usize, stride: usize, normalizer: C) -> Self {
        Self {
            pool_h,
            pool_w,
            stride,
            normalizer,
            _variable: PhantomData,
        }
    }

    pub fn calculate_output_size(
        &self,
        (batch_size, in_channels, in_height, in_width): (usize, usize, usize, usize),
    ) -> (usize, usize, usize, usize) {
        assert_eq!(in_height % (self.pool_h), 0);
        assert_eq!(in_width % (self.pool_w), 0);

        let out_height = (in_height - self.pool_h) / self.stride + 1;
        let out_width = (in_width - self.pool_w) / self.stride + 1;
        let out_channels = in_channels;
        (batch_size, out_channels, out_height, out_width)
    }

    pub fn avg_pool_naive(&self, input: &Input<F>, out: &mut Output<F>) {
        let (batch_size, in_channels, in_height, in_width) = input.dim();
        assert_eq!(in_height % (self.pool_h), 0);
        assert_eq!(in_width % (self.pool_w), 0);

        let out_dim = self.calculate_output_size(input.dim());
        assert_eq!(out.dim(), out_dim);
        for b_i in 0..batch_size {
            for (out_i, i) in (0..in_height).step_by(self.stride).enumerate() {
                for (out_j, j) in (0..in_width).step_by(self.stride).enumerate() {
                    for chan in 0..in_channels {
                        let mut sum = F::zero();
                        for k_i in 0..self.pool_h {
                            for k_j in 0..self.pool_w {
                                sum += input[(b_i, chan, (i + k_i), (j + k_j))];
                            }
                        }
                        out[(b_i, chan, out_i, out_j)] = sum * self.normalizer;
                    }
                }
            }
        }
    }
}
