use rand_chacha::ChaChaRng;
use rand_core::SeedableRng;
use std::{marker::PhantomData, ops::Neg};

use crate::additive_share::{AdditiveShare, Share};
use algebra::{
    fixed_point::{FixedPoint, FixedPointParameters},
    PrimeField, UniformRandom,
};
use serde::{Deserialize, Serialize};

/// Shares of a triple `[[a]]`, `[[b]]`, `[[c]]` such that `ab = c`.
#[derive(Serialize, Deserialize, Copy, Clone)]
#[serde(bound = "P: PrimeField")]
pub struct Triple<P: PrimeField> {
    /// A share of the `a` part of the triple.
    pub a: P,
    /// A share of the `b` part of the triple.
    pub b: P,
    /// A share of the `c` part of the triple.
    pub c: P,
}

/// Shares of the intermediate step.
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(bound = "T: Share")]
pub struct BlindedSharedInputs<T: Share> {
    /// A share of the `x-a`.
    pub blinded_x: AdditiveShare<T>,
    /// A share of the `y-b`.
    pub blinded_y: AdditiveShare<T>,
}

/// Result of combining shares in `BlindedSharedInput`.
#[derive(Serialize, Deserialize)]
#[serde(bound = "T: Share")]
pub struct BlindedInputs<T: Share> {
    /// `x-a`.
    pub blinded_x: T,
    /// `y-b`.
    pub blinded_y: T,
}

/// Objects that can be multiplied via Beaver's triples protocols must implement
/// this trait.
pub trait BeaversMul<T>
where
    T: Share,
    <T as Share>::Ring: PrimeField,
{
    /// Share inputs by consuming a triple.
    fn share_and_blind_inputs(
        x: &AdditiveShare<T>,
        y: &AdditiveShare<T>,
        triple: &Triple<T::Ring>,
    ) -> BlindedSharedInputs<T> {
        let blinded_x = T::randomize_local_share(x, &triple.a.neg());
        let blinded_y = T::randomize_local_share(y, &triple.b.neg());
        BlindedSharedInputs {
            blinded_x,
            blinded_y,
        }
    }

    /// Reconstruct inputs that have been blinded in the previous step.
    fn reconstruct_blinded_inputs(
        b1: BlindedSharedInputs<T>,
        b2: BlindedSharedInputs<T>,
    ) -> BlindedInputs<T> {
        BlindedInputs {
            blinded_x: b1.blinded_x.combine(&b2.blinded_x),
            blinded_y: b1.blinded_y.combine(&b2.blinded_y),
        }
    }

    /// Multiply blinded inputs.
    fn multiply_blinded_inputs(
        party_index: usize,
        bl: BlindedInputs<T>,
        t: &Triple<T::Ring>,
    ) -> AdditiveShare<T>;
}

/// An implementation of Beaver's multiplication algorithm for shares of `FixedPoint<P>`.
pub struct FPBeaversMul<P: FixedPointParameters>(PhantomData<P>);

impl<P: FixedPointParameters> BeaversMul<FixedPoint<P>> for FPBeaversMul<P> {
    fn multiply_blinded_inputs(
        party_index: usize,
        bl: BlindedInputs<FixedPoint<P>>,
        t: &Triple<P::Field>,
    ) -> AdditiveShare<FixedPoint<P>> {
        let alpha = bl.blinded_x.inner;
        let beta = bl.blinded_y.inner;
        let res = if party_index == 1 {
            t.c + (alpha * t.b) + (beta * t.a) + (alpha * beta)
        } else {
            t.c + (alpha * t.b) + (beta * t.a)
        };
        AdditiveShare::new(FixedPoint::with_num_muls(res, 1))
    }
}

/// An **insecure** method of generating triples. This is intended *purely* for
/// testing purposes.
pub struct InsecureTripleGen<T: Share>(ChaChaRng, PhantomData<T>);

impl<T: Share> InsecureTripleGen<T> {
    /// Create a new `Self` from a random seed.
    pub fn new(seed: [u8; 32]) -> Self {
        Self(ChaChaRng::from_seed(seed), PhantomData)
    }
}

impl<T: Share> InsecureTripleGen<T>
where
    <T as Share>::Ring: PrimeField,
{
    /// Sample a triple for both parties.
    pub fn generate_triple_shares(&mut self) -> (Triple<T::Ring>, Triple<T::Ring>) {
        let a = T::Ring::uniform(&mut self.0);
        let b = T::Ring::uniform(&mut self.0);
        let c = a * b;
        let a_randomizer = T::Ring::uniform(&mut self.0);
        let b_randomizer = T::Ring::uniform(&mut self.0);
        let c_randomizer = T::Ring::uniform(&mut self.0);
        let party_1_triple = Triple {
            a: a - a_randomizer,
            b: b - b_randomizer,
            c: c - c_randomizer,
        };

        let party_2_triple = Triple {
            a: a_randomizer,
            b: b_randomizer,
            c: c_randomizer,
        };
        (party_1_triple, party_2_triple)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use algebra::fields::near_mersenne_64::F;
    use rand::Rng;

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
        let mul = if is_neg { -10.0 } else { 10.0 };
        let float: f64 = rng.gen();
        let f = TenBitExpFP::truncate_float(float * mul);
        let n = TenBitExpFP::from(f);
        (f, n)
    }

    #[test]
    fn test_triple_gen() {
        let mut gen = InsecureTripleGen::<TenBitExpFP>::new(RANDOMNESS);
        for _ in 0..1000 {
            let (t1, t2) = gen.generate_triple_shares();
            assert_eq!((t1.a + &t2.a) * (t1.b + &t2.b), (t1.c + &t2.c));
        }
    }

    #[test]
    fn test_share_and_blind() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let seed = RANDOMNESS;
        let mut gen = InsecureTripleGen::<TenBitExpFP>::new(seed);
        for _ in 0..1000 {
            let (t1, t2) = gen.generate_triple_shares();
            let (_, n1) = generate_random_number(&mut rng);
            let (_, n2) = generate_random_number(&mut rng);
            let (s11, s12) = n1.share(&mut rng);
            let (s21, s22) = n2.share(&mut rng);
            let p1_bl_input = FPBeaversMul::share_and_blind_inputs(&s11, &s21, &t1);
            let p2_bl_input = FPBeaversMul::share_and_blind_inputs(&s12, &s22, &t2);
            let a = t1.a + &t2.a;
            let b = t1.b + &t2.b;
            assert_eq!(
                p1_bl_input.blinded_x.combine(&p2_bl_input.blinded_x),
                n1 - TenBitExpFP::new(a)
            );
            assert_eq!(
                p1_bl_input.blinded_y.combine(&p2_bl_input.blinded_y),
                n2 - TenBitExpFP::new(b)
            );
        }
    }

    #[test]
    fn test_reconstruct_blinded_inputs() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let seed = RANDOMNESS;
        let mut gen = InsecureTripleGen::<TenBitExpFP>::new(seed);
        for _ in 0..1000 {
            let (t1, t2) = gen.generate_triple_shares();
            let (_, n1) = generate_random_number(&mut rng);
            let (_, n2) = generate_random_number(&mut rng);
            let (s11, s12) = n1.share(&mut rng);
            let (s21, s22) = n2.share(&mut rng);
            let p1_bl_input = FPBeaversMul::share_and_blind_inputs(&s11, &s21, &t1);
            let p2_bl_input = FPBeaversMul::share_and_blind_inputs(&s12, &s22, &t2);
            let (p1_bl_input, p2_bl_input) = (
                FPBeaversMul::reconstruct_blinded_inputs(p1_bl_input, p2_bl_input),
                FPBeaversMul::reconstruct_blinded_inputs(p2_bl_input, p1_bl_input),
            );
            let a = t1.a + &t2.a;
            let b = t1.b + &t2.b;
            assert_eq!(p1_bl_input.blinded_x, p2_bl_input.blinded_x);
            assert_eq!(p1_bl_input.blinded_x, n1 - TenBitExpFP::new(a));
            assert_eq!(p1_bl_input.blinded_y, p2_bl_input.blinded_y);
            assert_eq!(p1_bl_input.blinded_y, n2 - TenBitExpFP::new(b));
        }
    }

    #[test]
    fn test_beavers_mul() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let seed = RANDOMNESS;
        let mut gen = InsecureTripleGen::<TenBitExpFP>::new(seed);
        for _ in 0..1000 {
            let (t1, t2) = gen.generate_triple_shares();
            let (f1, n1) = generate_random_number(&mut rng);
            let (f2, n2) = generate_random_number(&mut rng);
            let f3 = f1 * f2;
            let n3 = TenBitExpFP::from(f3);
            let (s11, s12) = n1.share(&mut rng);
            let (s21, s22) = n2.share(&mut rng);

            let p1_bl_input = FPBeaversMul::share_and_blind_inputs(&s11, &s21, &t1);
            let p2_bl_input = FPBeaversMul::share_and_blind_inputs(&s12, &s22, &t2);
            let (p1_bl_input, p2_bl_input) = (
                FPBeaversMul::reconstruct_blinded_inputs(p1_bl_input, p2_bl_input),
                FPBeaversMul::reconstruct_blinded_inputs(p2_bl_input, p1_bl_input),
            );

            let s31 = FPBeaversMul::multiply_blinded_inputs(1, p1_bl_input, &t1);
            let s32 = FPBeaversMul::multiply_blinded_inputs(2, p2_bl_input, &t2);
            let n4 = s31.combine(&s32);
            assert_eq!(
                n4, n3,
                "test failed with f1 = {:?}, f2 = {:?}, f3 = {:?}",
                f1, f2, f3
            );
        }
    }

    #[test]
    fn test_beavers_mul_with_trunc() {
        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let seed = RANDOMNESS;
        let mut gen = InsecureTripleGen::<TenBitExpFP>::new(seed);
        for _ in 0..1000 {
            let (t1, t2) = gen.generate_triple_shares();
            let (f1, n1) = generate_random_number(&mut rng);
            let (f2, n2) = generate_random_number(&mut rng);
            let f3 = f1 * f2;
            let n3 = TenBitExpFP::from(f3);
            let (s11, s12) = n1.share(&mut rng);
            let (s21, s22) = n2.share(&mut rng);

            let p1_bl_input = FPBeaversMul::share_and_blind_inputs(&s11, &s21, &t1);
            let p2_bl_input = FPBeaversMul::share_and_blind_inputs(&s12, &s22, &t2);
            let (p1_bl_input, p2_bl_input) = (
                FPBeaversMul::reconstruct_blinded_inputs(p1_bl_input, p2_bl_input),
                FPBeaversMul::reconstruct_blinded_inputs(p2_bl_input, p1_bl_input),
            );

            let mut s31 = FPBeaversMul::multiply_blinded_inputs(1, p1_bl_input, &t1);
            let mut s32 = FPBeaversMul::multiply_blinded_inputs(2, p2_bl_input, &t2);
            s31.inner.signed_reduce_in_place();
            s32.inner.signed_reduce_in_place();
            let n4 = s31.combine(&s32);
            assert_eq!(
                n4, n3,
                "test failed with f1 = {:?}, f2 = {:?}, f3 = {:?}",
                f1, f2, f3
            );
        }
    }
}
