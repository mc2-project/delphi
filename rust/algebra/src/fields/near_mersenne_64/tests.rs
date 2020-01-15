#[cfg(test)]
mod tests {
    use crate::fields::{
        tests::{field_test, primefield_test},
        UniformRandom,
    };
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;

    const RANDOMNESS: [u8; 32] = [
        0x99, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4,
        0x76, 0x5d, 0xc9, 0x8d, 0x62, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a,
        0x52, 0xd2,
    ];

    #[test]
    fn test_f() {
        use crate::fields::near_mersenne_64::F;

        let mut rng = ChaChaRng::from_seed(RANDOMNESS);

        let a = F::uniform(&mut rng);
        let b = F::uniform(&mut rng);
        field_test(a, b);
        primefield_test::<F>();
    }
}
