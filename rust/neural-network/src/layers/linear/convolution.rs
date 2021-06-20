use crate::tensors::{Input, Kernel, Output};
use algebra::{fp_64::Fp64Parameters, FixedPoint, FixedPointParameters, FpParameters, PrimeField};
use num_traits::Zero;
use std::{
    marker::PhantomData,
    ops::{AddAssign, Mul},
};
use tch::nn;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Padding {
    Same,
    Valid,
}

#[derive(Debug)]
pub struct Conv2dParams<F, C> {
    pub padding: Padding,
    pub stride: usize,
    pub kernel: Kernel<C>,
    pub bias: Kernel<C>,
    pub tch_config: Option<nn::Conv2D>,
    pub eval_method: crate::EvalMethod,
    _variable: PhantomData<F>,
}

unsafe impl<F, C> Send for Conv2dParams<F, C> {}
unsafe impl<F, C> Sync for Conv2dParams<F, C> {}

impl<F, C> Conv2dParams<F, C>
where
    F: Zero + Copy + Mul<C, Output = F> + AddAssign,
    C: Copy + Into<F>,
{
    pub fn new(padding: Padding, stride: usize, kernel: Kernel<C>, bias: Kernel<C>) -> Self {
        // Check whether the bias dimension are correct - it should have one element per
        // out_chan
        let kernel_dims = kernel.dim();
        let bias_dims = bias.dim();
        assert!(
            (bias_dims.0 == kernel_dims.0)
                && (bias_dims.1 == 1)
                && (bias_dims.2 == 1)
                && (bias_dims.3 == 1)
        );
        Self {
            padding,
            stride,
            kernel,
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
        let (num, k_channels, k_height, _) = self.kernel.dim();
        let padding = match self.padding {
            Padding::Same => (k_height - 1) / 2,
            Padding::Valid => 0,
        };
        assert_eq!(k_channels, in_channels);
        let k_size = k_height;
        let out_height = (in_height - k_size + 2 * padding) / self.stride + 1;
        let out_width = (in_width - k_size + 2 * padding) / self.stride + 1;
        let out_channels = num;
        (batch_size, out_channels, out_height, out_width)
    }

    #[inline]
    fn get_with_padding(
        &self,
        input: &Input<F>,
        batch: usize,
        mut row: isize,
        mut col: isize,
        chan: usize,
        p: u8,
    ) -> F {
        let (_, _, in_height, in_width) = input.dim();
        let in_height = in_height as isize;
        let in_width = in_width as isize;
        row -= p as isize;
        col -= p as isize;
        if row < 0 || col < 0 || row >= in_height || col >= in_width {
            F::zero()
        } else {
            unsafe { *input.uget((batch, chan, row as usize, col as usize)) }
        }
    }

    pub fn conv2d_naive(&self, input: &Input<F>, out: &mut Output<F>) {
        let (batch_size, in_channels, in_height, in_width) = input.dim();
        let (num, k_channels, k_height, k_width) = self.kernel.dim();
        assert_eq!(k_channels, in_channels);
        let p = match self.padding {
            Padding::Same => (k_height - 1) / 2,
            Padding::Valid => 0,
        };
        let out_dim = self.calculate_output_size(input.dim());
        // Check that we will always be in bounds during access:
        assert_eq!(out.dim(), out_dim);
        let ((in_h_start, in_h_end), (in_w_start, in_w_end)) = match self.padding {
            Padding::Same => ((0, in_height), (0, in_width)),
            Padding::Valid => ((0, out_dim.2), (0, out_dim.3)),
        };
        for b_i in 0..batch_size {
            for (out_i, i) in (in_h_start..in_h_end).step_by(self.stride).enumerate() {
                for (out_j, j) in (in_w_start..in_w_end).step_by(self.stride).enumerate() {
                    for in_chan in 0..in_channels {
                        for num_f in 0..num {
                            let mut sum = F::zero();
                            for k_i in 0..k_height {
                                for k_j in 0..k_width {
                                    let in_ij = self.get_with_padding(
                                        input,
                                        b_i,
                                        (i + k_i) as isize,
                                        (j + k_j) as isize,
                                        in_chan,
                                        p as u8,
                                    );
                                    let k_ij =
                                        unsafe { *self.kernel.uget((num_f, in_chan, k_i, k_j)) };
                                    sum += in_ij * k_ij;
                                }
                            }
                            unsafe {
                                *out.uget_mut((b_i, num_f, out_i, out_j)) += sum;
                            }
                        }
                    }
                }
            }
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
}

impl<P: FixedPointParameters, I> Conv2dParams<I, FixedPoint<P>>
where
    I: Zero + Copy + Into<FixedPoint<P>> + AddAssign + Mul<FixedPoint<P>, Output = I>,
    FixedPoint<P>: Into<I>,
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    pub fn new_with_gpu(
        vs: &nn::Path,
        padding: Padding,
        stride: usize,
        kernel: Kernel<FixedPoint<P>>,
        mut bias: Kernel<FixedPoint<P>>,
    ) -> Self {
        let (out_channels, in_channels, k_h, _) = kernel.dim();
        let device = vs.device();
        let kernel_tensor = kernel.to_tensor().to_device(device);
        let one = FixedPoint::one();
        for b in &mut bias {
            *b *= one;
        }
        let bias_tensor = bias
            .to_tensor()
            .reshape(&[out_channels as i64])
            .to_device(device);
        let mut out = Self::new(padding, stride, kernel, bias);
        out.eval_method = crate::EvalMethod::TorchDevice(device);

        assert_eq!(kernel_tensor.kind(), tch::Kind::Double);
        assert_eq!(bias_tensor.kind(), tch::Kind::Double);

        let p = match padding {
            Padding::Same => (k_h - 1) / 2,
            Padding::Valid => 0,
        };
        let conv2d_cfg = nn::ConvConfig {
            stride: stride as i64,
            padding: p as i64,
            bias: true,
            ..Default::default()
        };

        let mut tch_config = nn::conv2d(
            vs,
            in_channels as i64,
            out_channels as i64,
            k_h as i64,
            conv2d_cfg,
        );
        tch_config.ws = kernel_tensor;
        tch_config.bs = Some(bias_tensor);
        out.tch_config = Some(tch_config);
        out
    }
}
