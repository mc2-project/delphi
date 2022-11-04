#![allow(non_snake_case)]

use algebra::{BitIterator, FixedPointParameters, Fp64Parameters, FpParameters, PrimeField};
pub use fancy_garbling;

use fancy_garbling::{
    circuit::CircuitBuilder, error::CircuitBuilderError, util, BinaryBundle, BinaryGadgets,
    BundleGadgets, Fancy,
};

#[inline(always)]
fn mux_single_bit<F: Fancy>(
    f: &mut F,
    b: &F::Item,
    x: &F::Item,
    y: &F::Item,
) -> Result<F::Item, F::Error> {
    let y_plus_x = f.add(x, y)?;
    let res = f.mul(b, &y_plus_x)?;
    f.add(&x, &res)
}

/// If `b = 0` returns `x` else `y`.
///
/// `b` must be mod 2 but `x` and `y` can be have any modulus.
fn mux<F: Fancy>(
    f: &mut F,
    b: &F::Item,
    x: &BinaryBundle<F::Item>,
    y: &BinaryBundle<F::Item>,
) -> Result<Vec<F::Item>, F::Error> {
    x.wires()
        .iter()
        .zip(y.wires())
        .map(|(x, y)| mux_single_bit(f, b, x, y))
        .collect()
}

#[inline]
fn mod_p_helper<F: Fancy>(
    b: &mut F,
    neg_p: &BinaryBundle<F::Item>,
    bits: &BinaryBundle<F::Item>,
) -> Result<BinaryBundle<F::Item>, F::Error> {
    let (result, borrow) = b.bin_addition(&bits, &neg_p)?;
    // If p underflowed, then we want the result, otherwise we're fine with the
    // original.
    mux(b, &borrow, &bits, &result).map(BinaryBundle::new)
}

/// Binary adder. Returns the result and the carry.
fn adder_const<F: Fancy>(
    f: &mut F,
    x: &F::Item,
    y: &F::Item,
    b: bool,
    carry_in: Option<&F::Item>,
) -> Result<(F::Item, Option<F::Item>), F::Error> {
    if let Some(c) = carry_in {
        let z1 = f.xor(x, y)?;
        let z2 = f.xor(&z1, c)?;
        let z3 = f.xor(x, c)?;
        let z4 = f.and(&z1, &z3)?;
        let carry = f.xor(&z4, x)?;
        Ok((z2, Some(carry)))
    } else {
        let z = f.xor(x, y)?;
        let carry = if !b { None } else { Some(f.and(x, y)?) };
        Ok((z, carry))
    }
}

fn neg_p_over_2_helper<F: Fancy>(
    f: &mut F,
    neg_p_over_2: u128,
    neg_p_over_2_bits: &BinaryBundle<F::Item>,
    bits: &BinaryBundle<F::Item>,
) -> Result<BinaryBundle<F::Item>, F::Error> {
    let xwires = bits.wires();
    let ywires = neg_p_over_2_bits.wires();
    let mut neg_p_over_2 = BitIterator::new([neg_p_over_2 as u64]).collect::<Vec<_>>();
    neg_p_over_2.reverse();
    let mut neg_p_over_2 = neg_p_over_2.into_iter();
    let mut seen_one = neg_p_over_2.next().unwrap();

    let (mut z, mut c) = adder_const(f, &xwires[0], &ywires[0], seen_one, None)?;

    let mut bs = vec![z];
    for ((x, y), b) in xwires[1..(xwires.len() - 1)]
        .iter()
        .zip(&ywires[1..])
        .zip(neg_p_over_2)
    {
        seen_one |= b;
        let res = adder_const(f, x, y, seen_one, c.as_ref())?;
        z = res.0;
        c = res.1;
        bs.push(z);
    }

    z = f.add_many(&[
        xwires.last().unwrap().clone(),
        ywires.last().unwrap().clone(),
        c.unwrap(),
    ])?;
    bs.push(z);
    Ok(BinaryBundle::new(bs))
}

/// Compute the number of bits needed to represent `p`, plus one.
#[inline]
pub fn num_bits(p: u128) -> usize {
    (p.next_power_of_two() * 2).trailing_zeros() as usize
}

/// Compute the `ReLU` of `n` over the field `P::Field`.
pub fn relu<P: FixedPointParameters>(
    b: &mut CircuitBuilder,
    n: usize,
) -> Result<(), CircuitBuilderError>
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    let p = u128::from(<<P::Field as PrimeField>::Params>::MODULUS.0);
    let exponent_size = P::EXPONENT_CAPACITY as usize;

    let p_over_2 = p / 2;
    // Convert to two's complement
    let neg_p_over_2 = !p_over_2 + 1;
    // Convert to two's complement. Equivalent to `let neg_p = -(p as i128) as u128;
    let neg_p = !p + 1;
    let q = 2;
    let num_bits = num_bits(p);

    let moduli = vec![q; num_bits];
    // Construct constant for addition with neg p
    let neg_p = b.bin_constant_bundle(neg_p, num_bits)?;
    let neg_p_over_2_bits = b
        .constant_bundle(&util::u128_to_bits(neg_p_over_2, num_bits), &moduli)?
        .into();
    let zero = b.constant(0, 2)?;
    let one = b.constant(1, 2)?;
    for _ in 0..n {
        let s1 = BinaryBundle::new(b.evaluator_inputs(&moduli));
        let s2 = BinaryBundle::new(b.garbler_inputs(&moduli));
        let s2_next = BinaryBundle::new(b.garbler_inputs(&moduli));
        // Add secret shares as integers
        let res = b.bin_addition_no_carry(&s1, &s2)?;
        // Take the result mod p;
        let layer_input = mod_p_helper(b, &neg_p, &res).unwrap();

        // Compare with p/2
        // Since we take > p/2 as negative, if the number is less than p/2, it is
        // positive.
        let res = neg_p_over_2_helper(b, neg_p_over_2, &neg_p_over_2_bits, &layer_input)?;
        // Take the sign bit
        let zs_is_positive = res.wires().last().unwrap();

        // Compute the relu
        let mut relu_res = Vec::with_capacity(num_bits);
        let relu_6_size = exponent_size + 3;
        // We choose 5 arbitrarily here; the idea is that we won't see values of
        // greater than 2^8.
        // We then drop the larger bits
        for wire in layer_input.wires().iter().take(relu_6_size + 5) {
            relu_res.push(b.and(&zs_is_positive, wire)?);
        }
        let is_seven = b.and_many(&relu_res[(exponent_size + 1)..relu_6_size])?;
        let some_higher_bit_is_set = b.or_many(&relu_res[relu_6_size..])?;

        let should_be_six = b.or(&some_higher_bit_is_set, &is_seven)?;

        for wire in &mut relu_res[relu_6_size..] {
            *wire = zero;
        }
        let lsb = &mut relu_res[exponent_size];
        *lsb = mux_single_bit(b, &should_be_six, lsb, &zero)?;

        let middle_bit = &mut relu_res[exponent_size + 1];
        *middle_bit = mux_single_bit(b, &should_be_six, middle_bit, &one)?;

        let msb = &mut relu_res[exponent_size + 2];
        *msb = mux_single_bit(b, &should_be_six, msb, &one)?;

        for wire in &mut relu_res[..exponent_size] {
            *wire = mux_single_bit(b, &should_be_six, wire, &zero)?;
        }

        relu_res.extend(std::iter::repeat(zero).take(num_bits - relu_6_size - 5));

        let relu_res = BinaryBundle::new(relu_res);

        let res = b.bin_addition_no_carry(&relu_res, &s2_next)?;
        let next_share = mod_p_helper(b, &neg_p, &res)?;

        b.output_bundle(&next_share)?;
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Share;
    use algebra::{fields::near_mersenne_64::F, *};
    use fancy_garbling::circuit::CircuitBuilder;
    use rand::{thread_rng, Rng};

    struct TenBitExpParams {}
    impl FixedPointParameters for TenBitExpParams {
        type Field = F;
        const MANTISSA_CAPACITY: u8 = 3;
        const EXPONENT_CAPACITY: u8 = 10;
    }

    type TenBitExpFP = FixedPoint<TenBitExpParams>;

    fn generate_random_number<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
        let is_neg: bool = rng.gen();
        let mul = if is_neg { -10.0 } else { 10.0 };
        let float: f64 = rng.gen();
        let f = TenBitExpFP::truncate_float(float * mul);
        let n = TenBitExpFP::from(f);
        (f, n)
    }

    /// Compute the product of some u16s as a u128.
    #[inline]
    pub(crate) fn product(xs: &[u16]) -> u128 {
        xs.iter().fold(1, |acc, &x| acc * x as u128)
    }

    #[test]
    pub(crate) fn test_relu() {
        // TODO: There is currently an off-by-one in this test that causes it
        // to fail occasionally
        let mut rng = thread_rng();
        let n = 42;
        let q = 2;
        let p = <F as PrimeField>::Params::MODULUS.0 as u128;
        let Q = product(&vec![q; n]);
        println!("n={} q={} Q={}", n, q, Q);

        let mut b = CircuitBuilder::new();
        relu::<TenBitExpParams>(&mut b, 1).unwrap();
        let mut c = b.finish();
        let _ = c.print_info();

        let zero = TenBitExpFP::zero();
        let six = TenBitExpFP::from(6.0);
        for i in 0..10000 {
            let (_, n1) = generate_random_number(&mut rng);
            let (s1, s2) = n1.share(&mut rng);
            let res_should_be_fp = if n1 <= zero {
                zero
            } else if n1 > six {
                six
            } else {
                n1
            };
            let res_should_be = res_should_be_fp.inner.into_repr().0 as u128;
            let z1 = F::uniform(&mut rng).into_repr().0 as u128;
            let res_should_be = (res_should_be + z1) % p;

            let s1 = s1.inner.inner.into_repr().0 as u128;
            let mut garbler_inputs = util::u128_to_bits(s1, n);
            garbler_inputs.extend_from_slice(&util::u128_to_bits(z1, n));

            let s2 = s2.inner.inner.into_repr().0 as u128;
            let evaluator_inputs = util::u128_to_bits(s2, n);

            let (en, ev) = fancy_garbling::garble(&mut c).unwrap();
            let xs = en.encode_garbler_inputs(&garbler_inputs);
            let ys = en.encode_evaluator_inputs(&evaluator_inputs);
            let garbled_eval_results = ev.eval(&mut c, &xs, &ys).unwrap();

            let evaluated_results = c.eval_plain(&garbler_inputs, &evaluator_inputs).unwrap();
            assert!(
                util::u128_from_bits(&evaluated_results).abs_diff(res_should_be) <= 1,
                "Iteration {}, Pre-ReLU value is {}, value should be {}, {:?}",
                i,
                n1,
                res_should_be_fp,
                res_should_be_fp
            );
            assert!(
                util::u128_from_bits(&garbled_eval_results).abs_diff(res_should_be) <= 1,
                "Iteration {}, Pre-ReLU value is {}, value should be {}, {:?}",
                i,
                n1,
                res_should_be_fp,
                res_should_be_fp
            );
        }
    }
}
