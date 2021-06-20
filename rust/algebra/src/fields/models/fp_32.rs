use num_traits::{One, Zero};
use rand::{CryptoRng, RngCore};
use std::{
    cmp::{Ord, Ordering, PartialOrd},
    fmt::{Display, Formatter, Result as FmtResult},
    io::{Read, Result as IoResult, Write},
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    str::FromStr,
};

use serde::{Deserialize, Serialize};

use crate::{
    biginteger::{BigInteger as _BigInteger, BigInteger32 as BigInteger},
    bytes::{FromBytes, ToBytes},
    fields::{Field, FpParameters, PrimeField},
    UniformRandom,
};

pub trait Fp32Parameters: FpParameters<BigInt = BigInteger> {}

#[derive(Derivative, Serialize, Deserialize)]
#[derivative(
    Default(bound = "P: Fp32Parameters"),
    Hash(bound = "P: Fp32Parameters"),
    Clone(bound = "P: Fp32Parameters"),
    Copy(bound = "P: Fp32Parameters"),
    Debug(bound = "P: Fp32Parameters"),
    PartialEq(bound = "P: Fp32Parameters"),
    Eq(bound = "P: Fp32Parameters")
)]
#[serde(bound = "P: Fp32Parameters")]
/// Does *not* use Montgomery reduction.
pub struct Fp32<P: Fp32Parameters>(
    pub(crate) BigInteger,
    #[serde(skip)]
    #[derivative(Debug = "ignore")]
    PhantomData<P>,
);

impl<P: Fp32Parameters> Fp32<P> {
    #[inline]
    pub const fn new(element: BigInteger) -> Self {
        Fp32::<P>(element, PhantomData)
    }

    #[inline]
    fn is_valid(&self) -> bool {
        self.0 < P::MODULUS
    }

    #[inline]
    fn reduce(&mut self) {
        if !self.is_valid() {
            self.0.sub_noborrow(&P::MODULUS);
        }
    }
}

impl<P: Fp32Parameters> Zero for Fp32<P> {
    #[inline]
    fn zero() -> Self {
        Fp32::<P>(BigInteger::from(0), PhantomData)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<P: Fp32Parameters> One for Fp32<P> {
    #[inline]
    fn one() -> Self {
        Fp32::new(BigInteger::from(1))
    }

    #[inline]
    fn is_one(&self) -> bool {
        self == &Self::one()
    }
}

impl<P: Fp32Parameters> Field for Fp32<P> {
    #[inline]
    fn double(&self) -> Self {
        let mut temp = *self;
        temp.double_in_place();
        temp
    }

    #[inline]
    fn double_in_place(&mut self) -> &mut Self {
        // This cannot exceed the backing capacity.
        self.0.mul2();
        // However, it may need to be reduced.
        self.reduce();
        self
    }

    #[inline]
    fn square(&self) -> Self {
        let mut temp = self.clone();
        temp.square_in_place();
        temp
    }

    #[inline]
    fn square_in_place(&mut self) -> &mut Self {
        let cur = *self;
        *self *= &cur;
        self
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(self.pow(&[u64::from(P::MODULUS.0 - 2)]))
        }
    }

    fn inverse_in_place(&mut self) -> Option<&mut Self> {
        if let Some(inverse) = self.inverse() {
            *self = inverse;
            Some(self)
        } else {
            None
        }
    }

    #[inline]
    fn frobenius_map(&mut self, _: usize) {
        // No-op: No effect in a prime field.
    }
}

impl<P: Fp32Parameters> UniformRandom for Fp32<P> {
    /// Samples a uniformly random field element.
    #[inline]
    fn uniform<R: RngCore + CryptoRng>(r: &mut R) -> Self {
        loop {
            let mut tmp = Fp32::<P>(BigInteger::uniform(r), PhantomData);

            // Mask away the unused bits at the beginning.
            (&mut tmp.0).0 &= std::u32::MAX >> P::REPR_SHAVE_BITS;

            if tmp.is_valid() {
                return tmp;
            }
        }
    }
}

impl<P: Fp32Parameters> PrimeField for Fp32<P> {
    type Params = P;
    type BigInt = BigInteger;

    #[inline]
    fn from_repr(r: BigInteger) -> Self {
        let r = Fp32::new(r);
        if r.is_valid() {
            r
        } else {
            Self::zero()
        }
    }

    #[inline]
    fn into_repr(&self) -> BigInteger {
        self.0
    }

    #[inline]
    fn from_random_bytes(bytes: &[u8]) -> Option<Self> {
        let mut result = Self::zero();
        if result.0.read_le((&bytes[..]).by_ref()).is_ok() {
            (&mut result.0).0 &= std::u32::MAX >> P::REPR_SHAVE_BITS;
            if result.is_valid() {
                Some(result)
            } else {
                None
            }
        } else {
            None
        }
    }

    #[inline]
    fn multiplicative_generator() -> Self {
        Fp32::<P>(P::GENERATOR, PhantomData)
    }

    #[inline]
    fn root_of_unity() -> Self {
        Fp32::<P>(P::ROOT_OF_UNITY, PhantomData)
    }
}

impl<P: Fp32Parameters> ToBytes for Fp32<P> {
    #[inline]
    fn write<W: Write>(&self, writer: W) -> IoResult<()> {
        self.into_repr().write(writer)
    }
}

impl<P: Fp32Parameters> FromBytes for Fp32<P> {
    #[inline]
    fn read<R: Read>(reader: R) -> IoResult<Self> {
        BigInteger::read(reader).map(Fp32::from_repr)
    }
}

/// `Fp` elements are ordered lexicographically.
impl<P: Fp32Parameters> Ord for Fp32<P> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.into_repr().cmp(&other.into_repr())
    }
}

impl<P: Fp32Parameters> PartialOrd for Fp32<P> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<P: Fp32Parameters> FromStr for Fp32<P> {
    type Err = ();

    /// Interpret a string of numbers as a (congruent) prime field element.
    /// Does not accept unnecessary leading zeroes or a blank string.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err(());
        }

        if s == "0" {
            return Ok(Self::zero());
        }

        let mut res = Self::zero();

        let ten = Self::from_repr(<Self as PrimeField>::BigInt::from(10));

        let mut first_digit = true;

        for c in s.chars() {
            match c.to_digit(10) {
                Some(c) => {
                    if first_digit {
                        if c == 0 {
                            return Err(());
                        }

                        first_digit = false;
                    }

                    res.mul_assign(&ten);
                    res.add_assign(&Self::from_repr(<Self as PrimeField>::BigInt::from(
                        u64::from(c),
                    )));
                }
                None => {
                    return Err(());
                }
            }
        }
        if !res.is_valid() {
            Err(())
        } else {
            Ok(res)
        }
    }
}

impl<P: Fp32Parameters> Display for Fp32<P> {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "Fp32({})", self.0)
    }
}

impl<P: Fp32Parameters> Neg for Fp32<P> {
    type Output = Self;
    #[inline]
    #[must_use]
    fn neg(self) -> Self {
        if !self.is_zero() {
            let mut tmp = P::MODULUS;
            tmp.sub_noborrow(&self.0);
            Fp32::<P>(tmp, PhantomData)
        } else {
            self
        }
    }
}

impl<'a, P: Fp32Parameters> Add<&'a Fp32<P>> for Fp32<P> {
    type Output = Self;

    #[inline]
    fn add(self, other: &Self) -> Self {
        let mut result = self;
        result.add_assign(other);
        result
    }
}

impl<'a, P: Fp32Parameters> Sub<&'a Fp32<P>> for Fp32<P> {
    type Output = Self;

    #[inline]
    fn sub(self, other: &Self) -> Self {
        let mut result = self;
        result.sub_assign(other);
        result
    }
}

impl<'a, P: Fp32Parameters> Mul<&'a Fp32<P>> for Fp32<P> {
    type Output = Self;

    #[inline]
    fn mul(self, other: &Self) -> Self {
        let mut result = self;
        result.mul_assign(other);
        result
    }
}

impl<'a, P: Fp32Parameters> Div<&'a Fp32<P>> for Fp32<P> {
    type Output = Self;

    #[inline]
    fn div(self, other: &Self) -> Self {
        let mut result = self;
        result.mul_assign(&other.inverse().unwrap());
        result
    }
}

impl<'a, P: Fp32Parameters> AddAssign<&'a Self> for Fp32<P> {
    #[inline]
    fn add_assign(&mut self, other: &Self) {
        // This cannot exceed the backing capacity.
        self.0.add_nocarry(&other.0);
        // However, it may need to be reduced

        self.reduce();
    }
}

impl<'a, P: Fp32Parameters> SubAssign<&'a Self> for Fp32<P> {
    #[inline]
    fn sub_assign(&mut self, other: &Self) {
        // If `other` is larger than `self`, add the modulus to self first.
        if other.0 > self.0 {
            self.0.add_nocarry(&P::MODULUS);
        }

        self.0.sub_noborrow(&other.0);
    }
}

impl<'a, P: Fp32Parameters> MulAssign<&'a Self> for Fp32<P> {
    #[inline]
    fn mul_assign(&mut self, other: &Self) {
        let res = ((self.0).0 as u64) * ((other.0).0 as u64) % (P::MODULUS.0 as u64);
        (&mut self.0).0 = res as u32;
    }
}

impl<'a, P: Fp32Parameters> DivAssign<&'a Self> for Fp32<P> {
    #[inline]
    fn div_assign(&mut self, other: &Self) {
        self.mul_assign(&other.inverse().unwrap());
    }
}

impl_ops_traits!(Fp32, Fp32Parameters);
