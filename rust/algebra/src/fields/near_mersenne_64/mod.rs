use crate::{
    biginteger::BigInteger64 as BigInteger,
    fields::{Fp64, Fp64Parameters, FpParameters},
};

pub type F = Fp64<FParameters>;

pub struct FParameters;

impl Fp64Parameters for FParameters {}
impl FpParameters for FParameters {
    type BigInt = BigInteger;

    const MODULUS: BigInteger = BigInteger(2061584302081);

    const MODULUS_BITS: u32 = 41u32;

    const CAPACITY: u32 = Self::MODULUS_BITS - 1;

    const REPR_SHAVE_BITS: u32 = 23;

    const R: BigInteger = BigInteger(1099502679928);

    const R2: BigInteger = BigInteger(1824578462277);

    const INV: u64 = 2061584302079;

    const GENERATOR: BigInteger = BigInteger(7u64);

    const TWO_ADICITY: u32 = 37;

    const ROOT_OF_UNITY: BigInteger = BigInteger(624392905781);

    const MODULUS_MINUS_ONE_DIV_TWO: BigInteger = BigInteger(1030792151040);

    const T: BigInteger = BigInteger(15);

    const T_MINUS_ONE_DIV_TWO: BigInteger = BigInteger(7);
}

#[cfg(test)]
mod tests;
