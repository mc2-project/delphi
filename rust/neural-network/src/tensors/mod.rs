use algebra::{fp_64::Fp64Parameters, FixedPoint, FixedPointParameters, FpParameters, PrimeField};
use crypto_primitives::{AdditiveShare, Share};
use ndarray::Array4;
use num_traits::Zero;
use rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use tch::Tensor;

#[macro_use]
mod macros;

type Quadruple = (usize, usize, usize, usize);

ndarray_impl!(Input, Array4, Quadruple);
ndarray_impl!(Kernel, Array4, Quadruple);

pub type Output<F> = Input<F>;

impl<T: Share> Input<T> {
    pub fn share<R: RngCore + CryptoRng>(
        &self,
        rng: &mut R,
    ) -> (Input<AdditiveShare<T>>, Input<AdditiveShare<T>>) {
        let mut share_1 = Vec::with_capacity(self.len());
        let mut share_2 = Vec::with_capacity(self.len());
        for inp in self.iter() {
            let (s1, s2) = inp.share(rng);
            share_1.push(s1);
            share_2.push(s2);
        }

        let s1 = Input::from_shape_vec(self.dim(), share_1).expect("Shapes should be same");
        let s2 = Input::from_shape_vec(self.dim(), share_2).expect("Shapes should be same");
        (s1, s2)
    }

    pub fn share_with_randomness(
        &self,
        r: &Input<T::Ring>,
    ) -> (Input<AdditiveShare<T>>, Input<AdditiveShare<T>>) {
        assert_eq!(r.dim(), self.dim());
        let mut share_1 = Vec::with_capacity(r.len());
        let mut share_2 = Vec::with_capacity(r.len());
        self.iter()
            .zip(r.iter())
            .map(|(elem, r)| T::share_with_randomness(elem, r))
            .for_each(|(s1, s2)| {
                share_1.push(s1);
                share_2.push(s2);
            });
        let s1 = Input::from_shape_vec(r.dim(), share_1).expect("Shapes should be same");
        let s2 = Input::from_shape_vec(r.dim(), share_2).expect("Shapes should be same");
        (s1, s2)
    }
}

impl<I> Input<I> {
    /// If the underlying elements of `Input` are `FixedPoint` elements having
    /// base field `Fp32` backed by a `u64` (thus allowing lazy reduction),
    /// then we can convert it to an equivalent `nn::Tensor` containing `i64`s.
    pub fn to_tensor<P>(&self) -> Tensor
    where
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
        I: Copy + Into<FixedPoint<P>>,
    {
        let (b, c, h, w) = self.dim();
        let mut result = vec![0.0; self.len()];
        let mut all_num_muls_are_zero = true;
        result.iter_mut().zip(self.iter()).for_each(|(e1, e2)| {
            let fp: FixedPoint<P> = Into::<FixedPoint<P>>::into(*e2);
            let big_int = fp.inner.into_repr();
            *e1 = if big_int >= <P::Field as PrimeField>::Params::MODULUS_MINUS_ONE_DIV_TWO {
                -((<P::Field as PrimeField>::Params::MODULUS.0 as i64) - big_int.0 as i64) as f64
            } else {
                big_int.0 as f64
            };
            all_num_muls_are_zero &= fp.num_muls() == 0;
        });
        assert!(all_num_muls_are_zero);
        let result = Tensor::of_slice(&result).reshape(&[b as i64, c as i64, h as i64, w as i64]);
        result
    }

    pub fn from_tensor<P>(tensor: Tensor) -> Option<Self>
    where
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
        I: Copy + From<FixedPoint<P>>,
    {
        let shape = tensor
            .size4()
            .ok()
            .map(|(b, c, h, w)| (b as usize, c as usize, h as usize, w as usize))?;
        let mut output: Vec<f64> = tensor.into();
        let output = output
            .iter_mut()
            .map(|e| {
                let e = *e as i64 % <P::Field as PrimeField>::Params::MODULUS.0 as i64;
                let reduced = if e < 0 {
                    ((<P::Field as PrimeField>::Params::MODULUS.0 as i64) + e as i64) as u64
                } else {
                    e as u64
                };
                FixedPoint::with_num_muls(P::Field::from_repr(reduced.into()), 1).into()
            })
            .collect::<Vec<_>>();

        Input::from_shape_vec(shape, output).ok()
    }
}

impl<T: Share> Input<AdditiveShare<T>> {
    pub fn randomize_local_share(&mut self, r: &Input<T::Ring>) {
        for (inp, r) in self.iter_mut().zip(r.iter()) {
            *inp = T::randomize_local_share(inp, r);
        }
    }

    pub fn combine(&self, other: &Input<AdditiveShare<T>>) -> Input<T> {
        assert_eq!(self.dim(), other.dim());
        let combined = self
            .iter()
            .zip(other.iter())
            .map(|(s1, s2)| AdditiveShare::combine(s1, s2))
            .collect();
        Input::from_shape_vec(self.dim(), combined).expect("Shapes should be same")
    }
}

impl<I> Kernel<I> {
    /// If the underlying elements of `Kernel` are `FixedPoint` elements having
    /// base field `Fp64` backed by a `u64` then we can convert to an equivalent
    /// 'nn::Tensor' containing `i64`s.
    pub fn to_tensor<P>(&self) -> Tensor
    where
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
        I: Copy + Into<FixedPoint<P>>,
    {
        let (c_out, c_in, h, w) = self.dim();
        let mut result = vec![0.0; self.len()];
        result.iter_mut().zip(self.iter()).for_each(|(e1, e2)| {
            let fp: FixedPoint<P> = Into::<FixedPoint<P>>::into(*e2);
            let big_int = fp.inner.into_repr();
            *e1 = if big_int >= <P::Field as PrimeField>::Params::MODULUS_MINUS_ONE_DIV_TWO {
                -((<P::Field as PrimeField>::Params::MODULUS.0 as i64) - big_int.0 as i64) as f64
            } else {
                big_int.0 as f64
            };
            *e1 = fp.inner.into_repr().0 as f64;
        });
        Tensor::of_slice(&result).reshape(&[c_out as i64, c_in as i64, h as i64, w as i64])
    }

    /// If the underlying elements of `Kernel` are `FixedPoint` elements having
    /// base field `Fp64` backed by a `u64` we can convert to Kernel<u64>
    pub fn to_repr<P>(&self) -> Kernel<u64>
    where
        P: FixedPointParameters,
        <P::Field as PrimeField>::Params: Fp64Parameters,
        P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
        I: Copy + Into<FixedPoint<P>>,
    {
        let mut kernel: Kernel<u64> = Kernel::zeros(self.dim());
        kernel.iter_mut().zip(self).for_each(|(e1, e2)| {
            let fp: FixedPoint<P> = Into::<FixedPoint<P>>::into(*e2);
            *e1 = fp.inner.into_repr().0;
        });
        kernel
    }
}
