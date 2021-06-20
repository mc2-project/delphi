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
    biginteger::{BigInteger as _BigInteger, BigInteger64 as BigInteger},
    bytes::{FromBytes, ToBytes},
    fields::{Field, FpParameters, PrimeField},
    UniformRandom,
};

pub trait Fp64Parameters: FpParameters<BigInt = BigInteger> {}

#[derive(Derivative, Serialize, Deserialize)]
#[derivative(
    Default(bound = "P: Fp64Parameters"),
    Hash(bound = "P: Fp64Parameters"),
    Clone(bound = "P: Fp64Parameters"),
    Copy(bound = "P: Fp64Parameters"),
    PartialEq(bound = "P: Fp64Parameters"),
    Eq(bound = "P: Fp64Parameters")
)]
#[serde(bound = "P: Fp64Parameters")]
pub struct Fp64<P: Fp64Parameters>(pub(crate) BigInteger, #[serde(skip)] PhantomData<P>);

impl<P: Fp64Parameters> Fp64<P> {
    #[inline]
    pub const fn new(element: BigInteger) -> Self {
        Fp64::<P>(element, PhantomData)
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

    #[inline]
    fn mont_reduce(&mut self, mul_result: u128) {
        let m = (mul_result as u64).wrapping_mul(P::INV) as u128;
        (self.0).0 = ((mul_result + m * u128::from(P::MODULUS.0)) >> 64) as u64;
        self.reduce();
    }
}

impl<P: Fp64Parameters> Zero for Fp64<P> {
    #[inline]
    fn zero() -> Self {
        Fp64::<P>(BigInteger::from(0), PhantomData)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<P: Fp64Parameters> One for Fp64<P> {
    #[inline]
    fn one() -> Self {
        Fp64::new(BigInteger::from(P::R))
    }

    #[inline]
    fn is_one(&self) -> bool {
        self == &Self::one()
    }
}

impl<P: Fp64Parameters> Field for Fp64<P> {
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
            Some(self.pow(&[P::MODULUS.0 - 2]))
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

impl<P: Fp64Parameters> UniformRandom for Fp64<P> {
    /// Samples a uniformly random field element.
    #[inline]
    fn uniform<R: RngCore + CryptoRng>(r: &mut R) -> Self {
        loop {
            let mut tmp = Fp64::<P>(BigInteger::uniform(r), PhantomData);

            // Mask away the unused bits at the beginning.
            (&mut tmp.0).0 &= std::u64::MAX >> P::REPR_SHAVE_BITS;

            if tmp.is_valid() {
                return tmp;
            }
        }
    }
}

impl<P: Fp64Parameters> PrimeField for Fp64<P> {
    type Params = P;
    type BigInt = BigInteger;

    #[inline]
    fn from_repr(r: BigInteger) -> Self {
        let mut r = Fp64(r, PhantomData);
        if r.is_valid() {
            r.mul_assign(&Fp64(P::R2, PhantomData));
            r
        } else {
            Self::zero()
        }
    }

    #[inline]
    fn into_repr(&self) -> BigInteger {
        let mut r = *self;
        r.mont_reduce((self.0).0 as u128);
        r.0
    }

    #[inline]
    fn from_random_bytes(bytes: &[u8]) -> Option<Self> {
        let mut result = Self::zero();
        if result.0.read_le((&bytes[..]).by_ref()).is_ok() {
            (result.0).0 &= 0xffffffffffffffff >> P::REPR_SHAVE_BITS;
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
        Fp64::<P>(P::GENERATOR, PhantomData)
    }

    #[inline]
    fn root_of_unity() -> Self {
        Fp64::<P>(P::ROOT_OF_UNITY, PhantomData)
    }
}

impl<P: Fp64Parameters> ToBytes for Fp64<P> {
    #[inline]
    fn write<W: Write>(&self, writer: W) -> IoResult<()> {
        self.into_repr().write(writer)
    }
}

impl<P: Fp64Parameters> FromBytes for Fp64<P> {
    #[inline]
    fn read<R: Read>(reader: R) -> IoResult<Self> {
        BigInteger::read(reader).map(Fp64::from_repr)
    }
}

/// `Fp` elements are ordered lexicographically.
impl<P: Fp64Parameters> Ord for Fp64<P> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.into_repr().cmp(&other.into_repr())
    }
}

impl<P: Fp64Parameters> PartialOrd for Fp64<P> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<P: Fp64Parameters> FromStr for Fp64<P> {
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

impl<P: Fp64Parameters> Display for Fp64<P> {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "Fp64({})", self.0)
    }
}

impl<P: Fp64Parameters> std::fmt::Debug for Fp64<P> {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let big_int = self.into_repr();
        if big_int.0 > (P::MODULUS_MINUS_ONE_DIV_TWO.0) {
            write!(f, "Fp64({})", (P::MODULUS.0 as i64) - (big_int.0 as i64))
        } else {
            write!(f, "Fp64({})", big_int.0)
        }
    }
}

impl<P: Fp64Parameters> Neg for Fp64<P> {
    type Output = Self;
    #[inline]
    #[must_use]
    fn neg(self) -> Self {
        if !self.is_zero() {
            let mut tmp = P::MODULUS;
            tmp.sub_noborrow(&self.0);
            Fp64::<P>(tmp, PhantomData)
        } else {
            self
        }
    }
}

impl<'a, P: Fp64Parameters> Add<&'a Fp64<P>> for Fp64<P> {
    type Output = Self;

    #[inline]
    fn add(self, other: &Self) -> Self {
        let mut result = self;
        result.add_assign(other);
        result
    }
}

impl<'a, P: Fp64Parameters> Sub<&'a Fp64<P>> for Fp64<P> {
    type Output = Self;

    #[inline]
    fn sub(self, other: &Self) -> Self {
        let mut result = self;
        result.sub_assign(other);
        result
    }
}

impl<'a, P: Fp64Parameters> Mul<&'a Fp64<P>> for Fp64<P> {
    type Output = Self;

    #[inline]
    fn mul(self, other: &Self) -> Self {
        let mut result = self;
        result.mul_assign(other);
        result
    }
}

impl<'a, P: Fp64Parameters> Div<&'a Fp64<P>> for Fp64<P> {
    type Output = Self;

    #[inline]
    fn div(self, other: &Self) -> Self {
        let mut result = self;
        result.mul_assign(&other.inverse().unwrap());
        result
    }
}

impl<'a, P: Fp64Parameters> AddAssign<&'a Self> for Fp64<P> {
    #[inline]
    fn add_assign(&mut self, other: &Self) {
        // This cannot exceed the backing capacity.
        self.0.add_nocarry(&other.0);
        // However, it may need to be reduced

        self.reduce();
    }
}

impl<'a, P: Fp64Parameters> SubAssign<&'a Self> for Fp64<P> {
    #[inline]
    fn sub_assign(&mut self, other: &Self) {
        // If `other` is larger than `self`, add the modulus to self first.
        if other.0 > self.0 {
            self.0.add_nocarry(&P::MODULUS);
        }

        self.0.sub_noborrow(&other.0);
    }
}

impl<'a, P: Fp64Parameters> MulAssign<&'a Self> for Fp64<P> {
    #[inline]
    fn mul_assign(&mut self, other: &Self) {
        let prod = (self.0).0 as u128 * (other.0).0 as u128;
        self.mont_reduce(prod);
    }
}

impl<'a, P: Fp64Parameters> DivAssign<&'a Self> for Fp64<P> {
    #[inline]
    fn div_assign(&mut self, other: &Self) {
        self.mul_assign(&other.inverse().unwrap());
    }
}

impl_ops_traits!(Fp64, Fp64Parameters);
