use rand_core::{CryptoRng, RngCore};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use algebra::{
    fixed_point::{FixedPoint, FixedPointParameters},
    UniformRandom,
};
use num_traits::Zero;
use serde::{Deserialize, Serialize};

/// Represents a type that can be additively shared.
pub trait Share:
    Sized
    + Clone
    + Copy
    + std::fmt::Debug
    + Eq
    + Serialize
    + for<'de> Deserialize<'de>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<<Self as Share>::Constant, Output = Self>
    + Add<<Self as Share>::Constant, Output = Self>
    + Neg<Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<<Self as Share>::Constant>
    + AddAssign<<Self as Share>::Constant>
{
    /// The underlying ring that the shares are created over.
    type Ring: for<'a> Add<&'a Self::Ring, Output = Self::Ring>
        + for<'a> Sub<&'a Self::Ring, Output = Self::Ring>
        + Copy
        + Zero
        + Neg<Output = Self::Ring>
        + UniformRandom;

    /// The underlying ring that the shares are created over.
    type Constant: Into<Self>;

    /// Create shares for `self`.
    fn share<R: RngCore + CryptoRng>(
        &self,
        rng: &mut R,
    ) -> (AdditiveShare<Self>, AdditiveShare<Self>) {
        let r = Self::Ring::uniform(rng);
        self.share_with_randomness(&r)
    }

    /// Create shares for `self` using randomness `r`.
    fn share_with_randomness(&self, r: &Self::Ring) -> (AdditiveShare<Self>, AdditiveShare<Self>);

    /// Randomize a share `s` with randomness `r`.
    fn randomize_local_share(s: &AdditiveShare<Self>, r: &Self::Ring) -> AdditiveShare<Self>;
}

#[derive(Default, Hash, Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(bound = "T: Share")]
#[must_use]
/// Represents an additive share of `T`.
pub struct AdditiveShare<T: Share> {
    /// The secret share.
    pub inner: T,
}

impl<T: Share> AdditiveShare<T> {
    /// Construct a new share from `inner`.
    #[inline]
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    /// Combine two additive shares to obtain the shared value.
    pub fn combine(&self, other: &Self) -> T {
        self.inner + other.inner
    }

    /// Add a constant to the share.
    #[inline]
    pub fn add_constant(mut self, other: T::Constant) -> Self {
        self.inner += other;
        self
    }

    /// Add a constant to the share in place..
    #[inline]
    pub fn add_constant_in_place(&mut self, other: T::Constant) {
        self.inner += other;
    }
}

impl<T: Share + Zero> Zero for AdditiveShare<T> {
    fn zero() -> Self {
        Self::new(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }
}

impl<P: FixedPointParameters> AdditiveShare<FixedPoint<P>> {
    /// Double the share.
    #[inline]
    pub fn double(&self) -> Self {
        let mut result = *self;
        result.inner.double_in_place();
        result
    }

    /// Double the share in place.
    #[inline]
    pub fn double_in_place(&mut self) -> &mut Self {
        self.inner.double_in_place();
        self
    }
}

/// Iterate over `self.inner` as `u64`s
pub struct ShareIterator<T: Share + IntoIterator<Item = u64>> {
    inner: <T as IntoIterator>::IntoIter,
}

impl<T: Share + IntoIterator<Item = u64>> Iterator for ShareIterator<T> {
    type Item = u64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T: Share + IntoIterator<Item = u64>> ExactSizeIterator for ShareIterator<T> {}

impl<T: Share + IntoIterator<Item = u64>> IntoIterator for AdditiveShare<T> {
    type Item = u64;
    type IntoIter = ShareIterator<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            inner: self.inner.into_iter(),
        }
    }
}

impl<T: Share + std::iter::FromIterator<u64>> std::iter::FromIterator<u64> for AdditiveShare<T> {
    /// Creates a FixedPoint from an iterator over limbs in little-endian order
    #[inline]
    fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
        Self::new(T::from_iter(iter))
    }
}

impl<T: Share> Add<Self> for AdditiveShare<T> {
    type Output = Self;

    #[inline]
    fn add(mut self, other: Self) -> Self {
        self.inner = self.inner + other.inner;
        self
    }
}

impl<T: Share> AddAssign<Self> for AdditiveShare<T> {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.inner += other.inner;
    }
}

impl<T: Share> Sub for AdditiveShare<T> {
    type Output = Self;

    #[inline]
    fn sub(mut self, other: Self) -> Self {
        self.inner -= other.inner;
        self
    }
}

impl<T: Share> SubAssign for AdditiveShare<T> {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.inner -= other.inner;
    }
}

impl<T: Share> Neg for AdditiveShare<T> {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self {
        self.inner = -self.inner;
        self
    }
}

impl<T: Share> Mul<T::Constant> for AdditiveShare<T> {
    type Output = Self;
    #[inline]
    fn mul(mut self, other: T::Constant) -> Self {
        self *= other;
        self
    }
}

impl<T: Share> MulAssign<T::Constant> for AdditiveShare<T> {
    #[inline]
    fn mul_assign(&mut self, other: T::Constant) {
        self.inner *= other;
    }
}

impl<T: Share> From<T> for AdditiveShare<T> {
    #[inline]
    fn from(other: T) -> Self {
        Self { inner: other }
    }
}

impl<P: FixedPointParameters> From<AdditiveShare<FixedPoint<P>>> for FixedPoint<P> {
    #[inline]
    fn from(other: AdditiveShare<FixedPoint<P>>) -> Self {
        other.inner
    }
}

/// Operations on shares mimic those of `FixedPoint<P>` itself.
/// This means that
/// * Multiplication by a constant does not automatically truncate the result;
/// * Addition, subtraction, and addition by a constant automatically
/// promote the result to have the correct number of multiplications (max(in1,
/// in2));
/// * `signed_reduce` behaves the same on `FixedPoint<P>` and
///   `AdditiveShare<FixedPoint<P>>`.
impl<P: FixedPointParameters> Share for FixedPoint<P> {
    type Ring = P::Field;
    type Constant = Self;

    #[inline]
    fn share_with_randomness(&self, r: &Self::Ring) -> (AdditiveShare<Self>, AdditiveShare<Self>) {
        let mut cur = *self;
        cur.inner += r;
        (AdditiveShare::new(cur), AdditiveShare::new(Self::new(-*r)))
    }

    #[inline]
    fn randomize_local_share(cur: &AdditiveShare<Self>, r: &Self::Ring) -> AdditiveShare<Self> {
        let mut cur = *cur;
        cur.inner.inner += r;
        cur
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use algebra::fields::near_mersenne_64::F;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaChaRng;

    struct TenBitExpParams {}
    impl FixedPointParameters for TenBitExpParams {
        type Field = F;
        const MANTISSA_CAPACITY: u8 = 5;
        const EXPONENT_CAPACITY: u8 = 5;
    }

    type TenBitExpFP = FixedPoint<TenBitExpParams>;
    // type FPShare = AdditiveShare<TenBitExpFP>;

    const RANDOMNESS: [u8; 32] = [
        0x99, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4,
        0x76, 0x5d, 0xc9, 0x8d, 0x62, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a,
        0x52, 0xd2,
    ];

    fn generate_random_number<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
        let is_neg: bool = rng.gen();
        let mul = if is_neg { -10.0 } else { 10.0 };
        let float: f64 = rng.gen();
        let f = TenBitExpFP::truncate_float(float * mul);
        let n = TenBitExpFP::from(f);
        (f, n)
    }

    #[test]
    fn test_share_combine() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let (_, n) = generate_random_number(&mut rng);
            let (s1, s2) = n.share(&mut rng);
            assert_eq!(s1.combine(&s2), n);
        }
    }

    #[test]
    fn test_double() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let (_, n) = generate_random_number(&mut rng);
            let (mut s1, mut s2) = n.share(&mut rng);
            s1.double_in_place();
            s2.double_in_place();
            assert_eq!(s1.combine(&s2), n.double());
        }
    }

    #[test]
    fn test_neg() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let (_, n) = generate_random_number(&mut rng);
            let (mut s1, mut s2) = n.share(&mut rng);
            s1 = -s1;
            s2 = -s2;
            assert_eq!(s1.combine(&s2), -n);
        }
    }

    #[test]
    fn test_mul_by_const() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let (_, n1) = generate_random_number(&mut rng);
            let (_, n2) = generate_random_number(&mut rng);
            let (mut s1, mut s2) = n1.share(&mut rng);
            s1 = s1 * n2;
            s2 = s2 * n2;
            assert_eq!(s1.combine(&s2), n1 * n2);
        }
    }

    #[test]
    fn test_mul_by_const_with_trunc() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        for _ in 0..1000 {
            let (_, n1) = generate_random_number(&mut rng);
            let (_, n2) = generate_random_number(&mut rng);
            let (mut s1, mut s2) = n1.share(&mut rng);
            s1 = s1 * n2;
            s2 = s2 * n2;
            s1.inner.signed_reduce_in_place();
            s2.inner.signed_reduce_in_place();
            assert_eq!(s1.combine(&s2), n1 * n2);
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
            let (s11, s12) = n1.share(&mut rng);
            let (s21, s22) = n2.share(&mut rng);
            let s31 = s11 + s21;
            let s32 = s12 + s22;
            assert_eq!(
                s31.combine(&s32),
                n3,
                "test failed with f1 = {:?}, f2 = {:?}, f3 = {:?}",
                f1,
                f2,
                f3
            );
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
            let (s11, s12) = n1.share(&mut rng);
            let (s21, s22) = n2.share(&mut rng);
            let s31 = s11 - s21;
            let s32 = s12 - s22;
            assert_eq!(
                s31.combine(&s32),
                n3,
                "test failed with f1 = {:?}, f2 = {:?}, f3 = {:?}",
                f1,
                f2,
                f3
            );
        }
    }
}
