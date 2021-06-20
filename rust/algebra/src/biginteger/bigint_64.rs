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
pub struct BigInteger64(pub u64);

impl BigInteger64 {
    pub fn new(value: u64) -> Self {
        BigInteger64(value)
    }
}

impl BigInteger for BigInteger64 {
    #[inline]
    fn add_nocarry(&mut self, other: &Self) -> bool {
        let res = (self.0 as u128) + (other.0 as u128);
        let carry = (res & (1 << 65)) == 1;
        *self = BigInteger64(res as u64);
        carry
    }

    #[inline]
    fn sub_noborrow(&mut self, other: &Self) -> bool {
        let tmp = (1u128 << 64) + u128::from(self.0) - u128::from(other.0);
        let borrow = if tmp >> 64 == 0 { 1 } else { 0 };
        *self = BigInteger64(tmp as u64);
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
        64 - self.0.leading_zeros()
    }

    #[inline]
    fn from_bits(bits: &[bool]) -> Self {
        assert!(bits.len() <= 64);
        let mut acc: u64 = 0;

        let mut bits = bits.to_vec();
        bits.reverse();

        for bit in bits {
            acc <<= 1;
            acc += bit as u64;
        }
        Self::new(acc)
    }

    #[inline]
    fn to_bits(&self) -> Vec<bool> {
        let mut res = Vec::with_capacity(64);
        for b in BitIterator::new([self.0]) {
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
        BigInteger64(rng.gen())
    }
}

impl std::iter::FromIterator<u64> for BigInteger64 {
    /// Creates a BigInteger from an iterator over limbs in little-endian order
    #[inline]
    fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
        let mut cur = Self::default();
        let next = iter.into_iter().next().unwrap();
        cur.0 = next;
        cur
    }
}

impl ToBytes for BigInteger64 {
    #[inline]
    fn write<W: Write>(&self, writer: W) -> IoResult<()> {
        self.0.write(writer)
    }
}

impl FromBytes for BigInteger64 {
    #[inline]
    fn read<R: Read>(reader: R) -> IoResult<Self> {
        u64::read(reader).map(Self::new)
    }
}

pub struct IntoIter64 {
    int: BigInteger64,
    iterated: bool,
}

impl Iterator for IntoIter64 {
    type Item = u64;
    fn next(&mut self) -> Option<Self::Item> {
        if self.iterated {
            None
        } else {
            self.iterated = true;
            Some(self.int.0)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, Some(1))
    }
}

impl ExactSizeIterator for IntoIter64 {}

impl IntoIterator for BigInteger64 {
    type Item = u64;
    type IntoIter = IntoIter64;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            int: self,
            iterated: false,
        }
    }
}

impl Display for BigInteger64 {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "0x")?;
        write!(f, "{:016x}", self.0)
    }
}

impl Ord for BigInteger64 {
    #[inline]
    fn cmp(&self, other: &Self) -> ::std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for BigInteger64 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<::std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl From<BigInteger64> for u64 {
    #[inline]
    fn from(val: BigInteger64) -> Self {
        val.0
    }
}

impl From<u64> for BigInteger64 {
    #[inline]
    fn from(val: u64) -> BigInteger64 {
        Self::new(val)
    }
}
