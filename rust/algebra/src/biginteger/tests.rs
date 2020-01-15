use crate::biginteger::BigInteger;
use rand::{self, CryptoRng, RngCore, SeedableRng};
use rand_chacha::ChaChaRng;

const RANDOMNESS: [u8; 32] = [
    0x99, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0x62, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];
fn biginteger_arithmetic_test<B: BigInteger>(a: B, b: B, zero: B) {
    // zero == zero
    assert_eq!(zero, zero);

    // zero.is_zero() == true
    assert_eq!(zero.is_zero(), true);

    // a == a
    assert_eq!(a, a);

    // a + 0 = a
    let mut a0_add = a.clone();
    a0_add.add_nocarry(&zero);
    assert_eq!(a0_add, a);

    // a - 0 = a
    let mut a0_sub = a.clone();
    a0_sub.sub_noborrow(&zero);
    assert_eq!(a0_sub, a);

    // a - a = 0
    let mut aa_sub = a.clone();
    aa_sub.sub_noborrow(&a);
    assert_eq!(aa_sub, zero);

    // a + b = b + a
    let mut ab_add = a.clone();
    ab_add.add_nocarry(&b);
    let mut ba_add = b.clone();
    ba_add.add_nocarry(&a);
    assert_eq!(ab_add, ba_add);
}

fn biginteger_bytes_test<B: BigInteger, R: RngCore + CryptoRng>(r: &mut R) {
    let mut bytes = [0u8; 256];
    let x = B::uniform(r);
    x.write(bytes.as_mut()).unwrap();
    let y = B::read(bytes.as_ref()).unwrap();
    assert_eq!(x, y);
}

fn test_biginteger<B: BigInteger, R: RngCore + CryptoRng>(zero: B, r: &mut R) {
    let a = B::uniform(r);
    let b = B::uniform(r);
    biginteger_arithmetic_test(a, b, zero);
    biginteger_bytes_test::<B, R>(r);
}

#[test]
fn test_biginteger32() {
    use crate::biginteger::BigInteger32 as B;
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    test_biginteger(B::new(0u32), &mut rng);
}

#[test]
fn test_biginteger64() {
    use crate::biginteger::BigInteger64 as B;
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    test_biginteger(B::new(0u64), &mut rng);
}

#[test]
fn test_biginteger128() {
    use crate::biginteger::BigInteger128 as B;
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    test_biginteger(B::new([0u64; 2]), &mut rng);
}

#[test]
fn test_biginteger256() {
    use crate::biginteger::BigInteger256 as B;
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    test_biginteger(B::new([0u64; 4]), &mut rng);
}

#[test]
fn test_biginteger384() {
    use crate::biginteger::BigInteger384 as B;
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    test_biginteger(B::new([0u64; 6]), &mut rng);
}
