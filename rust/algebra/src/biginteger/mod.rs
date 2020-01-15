use crate::{
    bytes::{FromBytes, ToBytes},
    fields::BitIterator,
};
use rand::{CryptoRng, Rng, RngCore};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display},
    io::{Read, Result as IoResult, Write},
    iter::FromIterator,
};

#[macro_use]
mod macros;

mod bigint_32;
pub use bigint_32::*;
mod bigint_64;
pub use bigint_64::*;

// bigint_impl!(BigInteger64, 1, IntoIter64);
bigint_impl!(BigInteger128, 2, IntoIter128);
bigint_impl!(BigInteger256, 4, IntoIter256);
bigint_impl!(BigInteger384, 6, IntoIter384);

#[cfg(test)]
mod tests;

/// This defines a `BigInteger`, a smart wrapper around a
/// sequence of `u64` limbs, least-significant digit first.
pub trait BigInteger:
    ToBytes
    + FromBytes
    + Copy
    + Clone
    + Debug
    + Default
    + Display
    + Eq
    + Ord
    + Send
    + Sized
    + Sync
    + 'static
    + From<u64>
    + IntoIterator<Item = u64>
    + FromIterator<u64>
{
    /// Add another representation to this one, returning the carry bit.
    fn add_nocarry(&mut self, other: &Self) -> bool;

    /// Subtract another representation from this one, returning the borrow bit.
    fn sub_noborrow(&mut self, other: &Self) -> bool;

    /// Performs a leftwise bitshift of this number, effectively multiplying
    /// it by 2. Overflow is ignored.
    fn mul2(&mut self);

    /// Performs a leftwise bitshift of this number by some amount.
    fn muln(&mut self, amt: u32);

    /// Performs a rightwise bitshift of this number, effectively dividing
    /// it by 2.
    fn div2(&mut self);

    /// Performs a rightwise bitshift of this number by some amount.
    fn divn(&mut self, amt: u32);

    /// Returns true iff this number is odd.
    fn is_odd(&self) -> bool;

    /// Returns true iff this number is even.
    fn is_even(&self) -> bool;

    /// Returns true iff this number is zero.
    fn is_zero(&self) -> bool;

    /// Compute the number of bits needed to encode this number. Always a
    /// multiple of 64.
    fn num_bits(&self) -> u32;

    /// Returns the big integer representation of a given big endian boolean
    /// array.
    fn from_bits(bits: &[bool]) -> Self;

    /// Returns the bit representation in a big endian boolean array, without
    /// leading zeros.
    fn to_bits(&self) -> Vec<bool>;

    /// Returns a vector for wnaf.
    fn find_wnaf(&self) -> Vec<i64>;

    /// Samples a uniformly random instance of `Self`.
    fn uniform<R: RngCore + CryptoRng>(rng: &mut R) -> Self;

    /// Writes this `BigInteger` as a big endian integer. Always writes
    /// `(num_bits` / 8) bytes.
    fn write_le<W: Write>(&self, writer: &mut W) -> IoResult<()> {
        self.write(writer)
    }

    /// Reads a big endian integer occupying (`num_bits` / 8) bytes into this
    /// representation.
    fn read_le<R: Read>(&mut self, reader: &mut R) -> IoResult<()> {
        *self = Self::read(reader)?;
        Ok(())
    }
}

pub mod arithmetic {
    /// Calculate a + b + carry, returning the sum and modifying the
    /// carry value.
    #[inline(always)]
    pub(crate) fn adc(a: u64, b: u64, carry: &mut u64) -> u64 {
        let tmp = u128::from(a) + u128::from(b) + u128::from(*carry);

        *carry = (tmp >> 64) as u64;

        tmp as u64
    }

    /// Calculate a - b - borrow, returning the result and modifying
    /// the borrow value.
    #[inline(always)]
    pub(crate) fn sbb(a: u64, b: u64, borrow: &mut u64) -> u64 {
        let tmp = (1u128 << 64) + u128::from(a) - u128::from(b) - u128::from(*borrow);

        *borrow = if tmp >> 64 == 0 { 1 } else { 0 };

        tmp as u64
    }

    /// Calculate a + (b * c) + carry, returning the least significant digit
    /// and setting carry to the most significant digit.
    #[inline(always)]
    pub(crate) fn mac_with_carry(a: u64, b: u64, c: u64, carry: &mut u64) -> u64 {
        let tmp = (u128::from(a)) + u128::from(b) * u128::from(c) + u128::from(*carry);

        *carry = (tmp >> 64) as u64;

        tmp as u64
    }
}
