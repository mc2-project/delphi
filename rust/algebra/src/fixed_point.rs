use derivative::Derivative;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Display, Formatter, Result as FmtResult},
    marker::PhantomData,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{
    biginteger::BigInteger,
    fields::{Field, FpParameters, PrimeField},
    UniformRandom,
};
use rand::{CryptoRng, RngCore};

/// `FixedPointParameters` represents the parameters for a fixed-point number.
/// `MANTISSA_CAPACITY + EXPONENT_CAPACITY` must be less than 64.
pub trait FixedPointParameters: Send + Sync {
    type Field: PrimeField;
    const MANTISSA_CAPACITY: u8;
    const EXPONENT_CAPACITY: u8;

    fn truncate_float(f: f64) -> f64 {
        let max_exp = f64::from(1 << Self::EXPONENT_CAPACITY);
        (f * max_exp).round() / max_exp
    }
}

#[derive(Derivative, Serialize, Deserialize)]
#[derivative(
    Default(bound = "P: FixedPointParameters"),
    Hash(bound = "P: FixedPointParameters"),
    Clone(bound = "P: FixedPointParameters"),
    Copy(bound = "P: FixedPointParameters"),
    Debug(bound = "P: FixedPointParameters"),
    Eq(bound = "P: FixedPointParameters")
)]
#[serde(bound = "P: FixedPointParameters")]
#[must_use]
pub struct FixedPoint<P: FixedPointParameters> {
    pub inner: P::Field,
    num_muls:  u8,
    #[serde(skip)]
    #[derivative(Debug = "ignore")]
    _params:   PhantomData<P>,
}

impl<P: FixedPointParameters> Zero for FixedPoint<P> {
    fn zero() -> Self {
        Self::zero()
    }

    fn is_zero(&self) -> bool {
        self.is_zero()
    }
}

impl<P: FixedPointParameters> One for FixedPoint<P> {
    fn one() -> Self {
        Self::one()
    }

    fn is_one(&self) -> bool {
        self.is_one()
    }
}

impl<P: FixedPointParameters> FixedPoint<P> {
    #[inline]
    pub fn new(inner: P::Field) -> Self {
        Self::with_num_muls(inner, 0)
    }

    #[inline]
    pub fn with_num_muls(inner: P::Field, num_muls: u8) -> Self {
        if Self::max_mul_capacity() < 1 {
            panic!(
                "cannot multiply or add because `P::MANTISSA_CAPACITY + P::EXPONENT_CAPACITY` is \
                 too large."
            );
        }
        Self {
            inner,
            num_muls,
            _params: PhantomData,
        }
    }

    #[inline]
    pub fn uniform<R: RngCore + CryptoRng>(r: &mut R) -> Self {
        Self::new(P::Field::uniform(r))
    }

    #[inline]
    pub fn truncate_float(f: f64) -> f64 {
        P::truncate_float(f)
    }

    #[inline]
    pub fn is_negative(&self) -> bool {
        let cur = self.inner.into_repr();
        let modulus_div_2 = <P::Field as PrimeField>::Params::MODULUS_MINUS_ONE_DIV_TWO;
        cur >= modulus_div_2
    }

    // We divide the space in half, with the top half representing negative numbers,
    // and the bottom half representing positive numbers.
    #[inline]
    const fn inner_capacity() -> u8 {
        (<P::Field as PrimeField>::Params::CAPACITY - 1) as u8
    }

    #[inline]
    const fn max_mul_capacity() -> u8 {
        Self::inner_capacity() / Self::size_in_bits()
    }

    #[inline]
    pub const fn num_muls(&self) -> u8 {
        self.num_muls
    }

    #[inline]
    const fn remaining_mul_capacity(&self) -> u8 {
        Self::max_mul_capacity() - self.num_muls
    }

    #[inline]
    pub const fn should_reduce(&self) -> bool {
        // Will multiplying further cause overflow?
        self.num_muls == Self::max_mul_capacity()
    }

    #[inline]
    pub const fn size_in_bits() -> u8 {
        P::MANTISSA_CAPACITY + P::EXPONENT_CAPACITY
    }

    #[inline]
    pub fn zero() -> Self {
        Self::new(P::Field::zero())
    }

    #[inline]
    pub fn one() -> Self {
        let mut one_repr = P::Field::one().into_repr();
        one_repr.muln(P::EXPONENT_CAPACITY as u32);
        let one = P::Field::from_repr(one_repr);
        Self {
            inner:    one,
            num_muls: 0,
            _params:  PhantomData,
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        *self == Self::one()
    }

    #[inline]
    pub fn double(&self) -> Self {
        let mut result = *self;
        result.double_in_place();
        result
    }

    #[inline]
    pub fn double_in_place(&mut self) -> &mut Self {
        self.inner.double_in_place();
        self
    }

    #[inline]
    pub fn signed_reduce(&self) -> Self {
        let mut result = *self;
        result.signed_reduce_in_place();
        result
    }

    #[inline]
    pub fn signed_reduce_in_place(&mut self) -> &mut Self {
        let cur_is_neg = self.is_negative();

        if cur_is_neg {
            *self = -*self;
        }
        self.reduce_in_place();
        if cur_is_neg {
            *self = -*self;
        }
        self
    }

    #[inline]
    pub fn signed_truncate_by(&self, by: usize) -> Self {
        let mut result = *self;
        result.signed_truncate_by_in_place(by);
        result
    }

    #[inline]
    pub fn signed_truncate_by_in_place(&mut self, by: usize) -> &mut Self {
        let cur_is_neg = self.is_negative();

        if cur_is_neg {
            *self = -*self;
        }
        self.truncate_in_place(by);
        if cur_is_neg {
            *self = -*self;
        }
        self
    }

    #[inline]
    fn reduce_in_place(&mut self) {
        self.truncate_in_place(usize::from(self.num_muls * P::EXPONENT_CAPACITY));
    }

    #[inline]
    fn truncate_in_place(&mut self, n_bits: usize) -> &mut Self {
        let mut repr = self.inner.into_repr();
        // We should shift down by the number of consumed bits, leaving only
        // the top (P::MANTISSA_CAPACITY + P::EXPONENT_CAPACITY) bits.
        repr.divn((n_bits) as u32);
        self.inner = P::Field::from_repr(repr);
        self.num_muls = 0;
        self
    }
}

pub fn discretized_cos<P: FixedPointParameters>(inp: FixedPoint<P>) -> FixedPoint<P> {
    let mut sum = FixedPoint::zero();
    for i in 0..4 {
        let mut res = inp.signed_truncate_by(i);
        for _ in 0..i {
            res.inner.double_in_place();
        }
        if i % 2 == 0 {
            sum.inner += &res.inner;
        } else {
            sum.inner -= &res.inner;
        }
    }
    // let mut smallest = FixedPoint::one();
    // smallest.truncate_in_place(usize::from(P::EXPONENT_CAPACITY - 1));
    // (sum.signed_truncate_by(7)).double()
    // sum
    // TODO: fix for arbitrary sizes (mostly wrt doubling)
    sum.signed_truncate_by(2)
}

impl<P: FixedPointParameters> Display for FixedPoint<P> {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}", f64::from(*self))
    }
}

impl<P: FixedPointParameters> Add for FixedPoint<P> {
    type Output = Self;

    #[inline]
    fn add(mut self, mut other: Self) -> Self {
        if self.num_muls > other.num_muls {
            let shift = (self.num_muls - other.num_muls) * P::EXPONENT_CAPACITY;
            for _ in 0..shift {
                other.double_in_place();
            }
            other.num_muls = self.num_muls;
            self + other
        } else if other.num_muls > self.num_muls {
            other + self
        } else {
            self.inner += &other.inner;
            self
        }
    }
}

impl<P: FixedPointParameters> AddAssign for FixedPoint<P> {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other
    }
}

impl<P: FixedPointParameters> Sub for FixedPoint<P> {
    type Output = Self;

    #[inline]
    fn sub(mut self, mut other: Self) -> Self {
        if self.num_muls > other.num_muls {
            let mut other_repr = other.inner.into_repr();
            let shift = (self.num_muls - other.num_muls) * P::EXPONENT_CAPACITY;
            other_repr.muln(shift.into());
            other = Self::new(P::Field::from_repr(other_repr));
            other.num_muls = self.num_muls;
            self - other
        } else if other.num_muls > self.num_muls {
            -(other - self)
        } else {
            self.inner -= &other.inner;
            self
        }
    }
}

impl<P: FixedPointParameters> SubAssign for FixedPoint<P> {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other
    }
}

impl<P: FixedPointParameters> Neg for FixedPoint<P> {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self {
        self.inner = -self.inner;
        self
    }
}

impl<P: FixedPointParameters> Mul for FixedPoint<P> {
    type Output = Self;

    #[inline]
    fn mul(mut self, mut other: Self) -> Self {
        if self.remaining_mul_capacity() > other.num_muls {
            self.inner *= &other.inner;
            self.num_muls += other.num_muls + 1;
            self
        } else if other.remaining_mul_capacity() > self.num_muls {
            other * self
        } else {
            // self.num_muls + other.num_muls > Self::max_mul_capacity()
            // That is, multiplying will overflow and wrap around.
            if other.remaining_mul_capacity() < self.remaining_mul_capacity() {
                other.signed_reduce_in_place();
                other * self
            } else {
                self.signed_reduce_in_place();
                self * other
            }
        }
    }
}

impl<P: FixedPointParameters> MulAssign for FixedPoint<P> {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other
    }
}

impl<P: FixedPointParameters> From<f64> for FixedPoint<P> {
    #[inline]
    fn from(other: f64) -> Self {
        if other.is_nan() {
            panic!("other is NaN: {:?}", other);
        }
        let val = (other.abs() * f64::from(1u32 << P::EXPONENT_CAPACITY)).round() as u64;
        let val = <P::Field as PrimeField>::BigInt::from(val);
        let mut val = P::Field::from_repr(val);
        if other.is_sign_negative() {
            val = -val
        }
        Self::new(val)
    }
}

impl<P: FixedPointParameters> From<f32> for FixedPoint<P> {
    #[inline]
    fn from(other: f32) -> Self {
        f64::from(other).into()
    }
}

impl<P: FixedPointParameters> From<FixedPoint<P>> for f64 {
    #[inline]
    fn from(mut other: FixedPoint<P>) -> Self {
        let is_negative = other.is_negative();
        if is_negative {
            other = -other
        }
        other.reduce_in_place();
        let inner = other.inner.into_repr();
        let mut inner = inner.into_iter();
        let len = inner.size_hint().0;
        let first = inner.next();

        if len == 1 || inner.all(|e| e == 0) {
            let ans = (first.unwrap() as f64) / f64::from(1 << P::EXPONENT_CAPACITY);
            if is_negative {
                -ans
            } else {
                ans
            }
        } else {
            panic!("other is too large to fit in f64 {:?}", other)
        }
    }
}

impl<P: FixedPointParameters> From<FixedPoint<P>> for f32 {
    #[inline]
    fn from(other: FixedPoint<P>) -> Self {
        f64::from(other) as f32
    }
}

pub struct FPIterator<F: FixedPointParameters> {
    int: <<F::Field as PrimeField>::BigInt as IntoIterator>::IntoIter,
}

impl<F: FixedPointParameters> Iterator for FPIterator<F> {
    type Item = u64;
    fn next(&mut self) -> Option<Self::Item> {
        self.int.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.int.size_hint()
    }
}

impl<F: FixedPointParameters> ExactSizeIterator for FPIterator<F> {}

impl<F: FixedPointParameters> IntoIterator for FixedPoint<F> {
    type Item = u64;
    type IntoIter = FPIterator<F>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            int: self.inner.into_repr().into_iter(),
        }
    }
}

impl<F: FixedPointParameters> std::iter::FromIterator<u64> for FixedPoint<F> {
    /// Creates a FixedPoint from an iterator over limbs in little-endian order
    #[inline]
    fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
        let big_int = <F::Field as PrimeField>::BigInt::from_iter(iter);
        Self::new(F::Field::from_repr(big_int))
    }
}

impl<P: FixedPointParameters> PartialEq for FixedPoint<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        let cur_is_neg = self.is_negative();
        let other_is_neg = other.is_negative();

        if cur_is_neg ^ other_is_neg {
            false
        } else {
            let mut cur = if cur_is_neg { -*self } else { *self };
            let mut other = if other_is_neg { -*other } else { *other };
            cur.reduce_in_place();
            other.reduce_in_place();
            let cur_repr = cur.inner.into_repr();
            let other_repr = other.inner.into_repr();
            let cur = cur_repr.into_iter().next().unwrap();
            let other = other_repr.into_iter().next().unwrap();
            cur.checked_sub(other)
                .map_or_else(|| (other - cur) <= 1, |res| res <= 1)
        }
    }
}

impl<P: FixedPointParameters> PartialOrd for FixedPoint<P> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        let cur_is_neg = self.is_negative();
        let other_is_neg = other.is_negative();

        if cur_is_neg && !other_is_neg {
            Some(Ordering::Less)
        } else if !cur_is_neg && other_is_neg {
            Some(Ordering::Greater)
        } else {
            let mut cur = if cur_is_neg { -*self } else { *self };
            let mut other = if other_is_neg { -*other } else { *other };
            cur.reduce_in_place();
            other.reduce_in_place();
            let cur_repr = cur.inner.into_repr();
            let other_repr = other.inner.into_repr();

            let cur = cur_repr.into_iter().next().unwrap();
            let other = other_repr.into_iter().next().unwrap();
            if cur
                .checked_sub(other)
                .map_or_else(|| (other - cur) <= 1, |res| res <= 1)
            {
                Some(Ordering::Equal)
            } else {
                cur_repr.partial_cmp(&other_repr)
            }
        }
    }
}

impl<P: FixedPointParameters> Ord for FixedPoint<P> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<P: FixedPointParameters> Into<u64> for FixedPoint<P>
where
    <P::Field as PrimeField>::Params: crate::Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    #[inline]
    fn into(self) -> u64 {
        self.inner.into_repr().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::near_mersenne_64::F;

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaChaRng;

    struct TenBitExpParams {}
    impl FixedPointParameters for TenBitExpParams {
        type Field = F;
        const MANTISSA_CAPACITY: u8 = 5;
        const EXPONENT_CAPACITY: u8 = 5;
    }

    type TenBitExpFP = FixedPoint<TenBitExpParams>;

    const RANDOMNESS: [u8; 32] = [
        0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4,
        0x76, 0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a,
        0x52, 0xd2,
    ];

    fn generate_random_number<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
        let is_neg: bool = rng.gen();
        let mut mul = if is_neg { -10.0 } else { 10.0 };
        let is_hundreds: bool = rng.gen();
        mul *= if is_hundreds { 10.0 } else { 1.0 };
        let float: f64 = rng.gen();
        let f = TenBitExpFP::truncate_float(float * mul);
        let n = TenBitExpFP::from(f);
        (f, n)
    }

    #[test]
    fn test_from_float() {
        let num_float = 17.5;
        let num_fixed = TenBitExpFP::from(num_float);
        let num_fixed_double = TenBitExpFP::from(num_float * 2.0);
        let one = TenBitExpFP::one();
        let two = one + one;
        let num_plus_one_fixed = TenBitExpFP::from(num_float + 1.0);
        let neg_num = TenBitExpFP::from(-num_float);
        assert_eq!(num_plus_one_fixed, one + num_fixed);
        assert_eq!(num_fixed + num_fixed, num_fixed_double);
        assert_eq!(num_fixed.double(), num_fixed_double);
        assert_eq!(num_fixed * two, num_fixed_double);
        assert_eq!(num_fixed + neg_num, TenBitExpFP::zero());
    }

    #[test]
    fn test_double() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let (f1, n1) = generate_random_number(&mut rng);
            let f2 = f1 * 2.0;
            let n2 = TenBitExpFP::from(f2);
            let error_msg = format!("test failed with f1 = {:?}, f2 = {:?}", f1, f2,);
            assert_eq!(n1.double(), n2, "{}", error_msg);
        }
    }

    #[test]
    fn test_is_negative() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let (f, n) = generate_random_number(&mut rng);
            let error_msg = format!("test failed with f = {:?}, n = {}", f, n);
            let f_is_neg = f.is_sign_negative() & (f != 0.0);
            assert_eq!(f_is_neg, n.is_negative(), "{}", error_msg);
        }
    }

    #[test]
    fn test_neg() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let (f1, n1) = generate_random_number(&mut rng);
            let f2 = -f1;
            let n2 = TenBitExpFP::from(f2);
            let error_msg = format!("test failed with f1 = {:?}, f2 = {:?}", f1, f2,);
            assert_eq!(-n1, n2, "{}", error_msg);
        }
    }

    #[test]
    fn test_add() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let (f1, n1) = generate_random_number(&mut rng);
            let (f2, n2) = generate_random_number(&mut rng);
            let f3 = f1 + f2;
            let n3 = TenBitExpFP::from(f3);
            let error_msg = format!(
                "test failed with f1 = {:?}, f2 = {:?}, f3 = {:?}",
                f1, f2, f3
            );
            assert_eq!(n1 + n2, n3, "{}", error_msg);
        }
    }

    #[test]
    fn test_sub() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let (f1, n1) = generate_random_number(&mut rng);
            let (f2, n2) = generate_random_number(&mut rng);
            let f3 = f1 - f2;
            let n3 = TenBitExpFP::from(f3);
            let error_msg = format!(
                "test failed with\nf1 = {:?}, f2 = {:?}, f3 = {:?}\nn1 = {}, n2 = {}, n3 = {}\n",
                f1, f2, f3, n1, n2, n3,
            );
            assert_eq!(n1 - n2, n3, "{}", error_msg);
        }
    }

    #[test]
    fn test_mul() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        for _ in 0..1000 {
            let (f1, n1) = generate_random_number(&mut rng);
            let (f2, n2) = generate_random_number(&mut rng);
            let f3 = TenBitExpFP::truncate_float(f1 * f2);
            let n3 = TenBitExpFP::from(f3);

            let error_msg = format!(
                "test failed with\nf1 = {:?}, f2 = {:?}, f3 = {:?}\nn1 = {}, n2 = {}, n3 = {}\n",
                f1, f2, f3, n1, n2, n3,
            );

            assert_eq!(n1 * n2, n3, "{}", error_msg);
        }
    }

    #[test]
    fn test_reduce_positive() {
        let one = TenBitExpFP::one();
        assert_eq!(one, one * one, "one is not equal to 1 * 1");
        let reduced_one = (one * one * one).signed_reduce();
        assert_eq!(one, reduced_one);
    }

    #[test]
    fn test_reduce_neg() {
        let one = TenBitExpFP::one();
        let neg_one = -one;
        assert_eq!(
            neg_one,
            one * neg_one,
            "negative one ({}) is not equal to -1 * 1 ({})",
            neg_one,
            one * neg_one,
        );
        let reduced_neg_one = (-one * one * one).signed_reduce();
        assert_eq!(neg_one, reduced_neg_one);
    }
}
