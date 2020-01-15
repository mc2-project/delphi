use crate::{
    biginteger::BigInteger,
    bytes::{FromBytes, ToBytes},
    fields::BitIterator,
};
use rand::{CryptoRng, Rng, RngCore};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Display,
    io::{Read, Result as IoResult, Write},
};

#[derive(Copy, Clone, PartialEq, Eq, Debug, Default, Hash, Serialize, Deserialize)]
pub struct BigInteger32(pub u32);

impl BigInteger32 {
    pub fn new(value: u32) -> Self {
        BigInteger32(value)
    }
}

impl BigInteger for BigInteger32 {
    #[inline]
    fn add_nocarry(&mut self, other: &Self) -> bool {
        let res = (self.0 as u64) + (other.0 as u64);
        let carry = (res & (1 << 33)) == 1;
        *self = BigInteger32(res as u32);
        carry
    }

    #[inline]
    fn sub_noborrow(&mut self, other: &Self) -> bool {
        let tmp = (1u64 << 32) + u64::from(self.0) - u64::from(other.0);
        let borrow = if tmp >> 32 == 0 { 1 } else { 0 };
        *self = BigInteger32(tmp as u32);
        borrow == 1
    }

    #[inline]
    fn mul2(&mut self) {
        *(&mut self.0) <<= 1;
    }

    #[inline]
    fn muln(&mut self, n: u32) {
        *(&mut self.0) <<= n;
    }

    #[inline]
    fn div2(&mut self) {
        *(&mut self.0) >>= 1;
    }

    #[inline]
    fn divn(&mut self, n: u32) {
        *(&mut self.0) >>= n;
    }

    #[inline]
    fn is_odd(&self) -> bool {
        self.0 & 1 == 1
    }

    #[inline]
    fn is_even(&self) -> bool {
        !self.is_odd()
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == 0
    }

    #[inline]
    fn num_bits(&self) -> u32 {
        32 - self.0.leading_zeros()
    }

    #[inline]
    fn from_bits(bits: &[bool]) -> Self {
        assert!(bits.len() <= 32);
        let mut acc: u32 = 0;

        let mut bits = bits.to_vec();
        bits.reverse();

        for bit in bits {
            acc <<= 1;
            acc += bit as u32;
        }
        Self::new(acc)
    }

    #[inline]
    fn to_bits(&self) -> Vec<bool> {
        let mut res = Vec::with_capacity(32);
        for b in BitIterator::new([self.0 as u64]) {
            res.push(b);
        }
        res
    }

    #[inline]
    fn find_wnaf(&self) -> Vec<i64> {
        vec![]
    }

    #[inline]
    fn uniform<R: RngCore + CryptoRng>(rng: &mut R) -> Self {
        BigInteger32(rng.gen())
    }
}

impl std::iter::FromIterator<u64> for BigInteger32 {
    /// Creates a BigInteger from an iterator over limbs in little-endian order
    #[inline]
    fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
        let mut cur = Self::default();
        let next = iter.into_iter().next().unwrap();
        cur.0 = next as u32;
        cur
    }
}

impl ToBytes for BigInteger32 {
    #[inline]
    fn write<W: Write>(&self, writer: W) -> IoResult<()> {
        self.0.write(writer)
    }
}

impl FromBytes for BigInteger32 {
    #[inline]
    fn read<R: Read>(reader: R) -> IoResult<Self> {
        u32::read(reader).map(Self::new)
    }
}

pub struct IntoIter32 {
    int:      BigInteger32,
    iterated: bool,
}

impl Iterator for IntoIter32 {
    type Item = u64;
    fn next(&mut self) -> Option<Self::Item> {
        if self.iterated {
            None
        } else {
            self.iterated = true;
            Some(self.int.0 as u64)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, Some(1))
    }
}

impl ExactSizeIterator for IntoIter32 {}

impl IntoIterator for BigInteger32 {
    type Item = u64;
    type IntoIter = IntoIter32;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            int:      self,
            iterated: false,
        }
    }
}

impl Display for BigInteger32 {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "0x")?;
        write!(f, "{:016x}", self.0)
    }
}

impl Ord for BigInteger32 {
    #[inline]
    fn cmp(&self, other: &Self) -> ::std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for BigInteger32 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<::std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl From<u64> for BigInteger32 {
    #[inline]
    fn from(val: u64) -> BigInteger32 {
        Self::new(val as u32)
    }
}
