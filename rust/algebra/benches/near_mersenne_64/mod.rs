use rand::SeedableRng;
use rand_chacha::ChaChaRng;

use algebra::{
    fields::near_mersenne_64::*, BigInteger as _, BigInteger64 as BigInteger, Field, PrimeField,
    UniformRandom,
};
use std::ops::{AddAssign, MulAssign, Neg, SubAssign};

const RANDOMNESS: [u8; 32] = [
    0x99, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8e, 0x63, 0x03, 0xf4, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

#[bench]
fn bench_f_repr_add_nocarry(b: &mut test::Bencher) {
    const SAMPLES: usize = 1000;

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    let v: Vec<(BigInteger, BigInteger)> = (0..SAMPLES)
        .map(|_| {
            let mut tmp1 = BigInteger::uniform(&mut rng);
            let mut tmp2 = BigInteger::uniform(&mut rng);
            // Shave a few bits off to avoid overflow.
            for _ in 0..3 {
                tmp1.div2();
                tmp2.div2();
            }
            (tmp1, tmp2)
        })
        .collect();

    let mut count = 0;
    b.iter(|| {
        let mut tmp = v[count].0;
        tmp.add_nocarry(&v[count].1);
        count = (count + 1) % SAMPLES;
        tmp
    });
}

#[bench]
fn bench_f_repr_sub_noborrow(b: &mut test::Bencher) {
    const SAMPLES: usize = 1000;

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    let v: Vec<(BigInteger, BigInteger)> = (0..SAMPLES)
        .map(|_| {
            let tmp1 = BigInteger::uniform(&mut rng);
            let mut tmp2 = tmp1;
            // Ensure tmp2 is smaller than tmp1.
            for _ in 0..10 {
                tmp2.div2();
            }
            (tmp1, tmp2)
        })
        .collect();

    let mut count = 0;
    b.iter(|| {
        let mut tmp = v[count].0;
        tmp.sub_noborrow(&v[count].1);
        count = (count + 1) % SAMPLES;
        tmp
    });
}

#[bench]
fn bench_f_repr_num_bits(b: &mut test::Bencher) {
    const SAMPLES: usize = 1000;

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    let v: Vec<BigInteger> = (0..SAMPLES)
        .map(|_| BigInteger::uniform(&mut rng))
        .collect();

    let mut count = 0;
    b.iter(|| {
        let tmp = v[count].num_bits();
        count = (count + 1) % SAMPLES;
        tmp
    });
}

#[bench]
fn bench_f_repr_mul2(b: &mut test::Bencher) {
    const SAMPLES: usize = 1000;

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    let v: Vec<BigInteger> = (0..SAMPLES)
        .map(|_| BigInteger::uniform(&mut rng))
        .collect();

    let mut count = 0;
    b.iter(|| {
        let mut tmp = v[count];
        tmp.mul2();
        count = (count + 1) % SAMPLES;
        tmp
    });
}

#[bench]
fn bench_f_repr_div2(b: &mut test::Bencher) {
    const SAMPLES: usize = 1000;

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    let v: Vec<BigInteger> = (0..SAMPLES)
        .map(|_| BigInteger::uniform(&mut rng))
        .collect();

    let mut count = 0;
    b.iter(|| {
        let mut tmp = v[count];
        tmp.div2();
        count = (count + 1) % SAMPLES;
        tmp
    });
}

#[bench]
fn bench_f_add_assign(b: &mut test::Bencher) {
    const SAMPLES: usize = 1000;

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    let v: Vec<(F, F)> = (0..SAMPLES)
        .map(|_| (F::uniform(&mut rng), F::uniform(&mut rng)))
        .collect();

    let mut count = 0;
    b.iter(|| {
        let mut tmp = v[count].0;
        tmp.add_assign(&v[count].1);
        count = (count + 1) % SAMPLES;
        tmp
    });
}

#[bench]
fn bench_f_sub_assign(b: &mut test::Bencher) {
    const SAMPLES: usize = 1000;

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    let v: Vec<(F, F)> = (0..SAMPLES)
        .map(|_| (F::uniform(&mut rng), F::uniform(&mut rng)))
        .collect();

    let mut count = 0;
    b.iter(|| {
        let mut tmp = v[count].0;
        tmp.sub_assign(&v[count].1);
        count = (count + 1) % SAMPLES;
        tmp
    });
}

#[bench]
fn bench_f_mul_assign(b: &mut test::Bencher) {
    const SAMPLES: usize = 1000;

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    let v: Vec<(F, F)> = (0..SAMPLES)
        .map(|_| (F::uniform(&mut rng), F::uniform(&mut rng)))
        .collect();

    let mut count = 0;
    b.iter(|| {
        let mut tmp = v[count].0;
        tmp.mul_assign(&v[count].1);
        count = (count + 1) % SAMPLES;
        tmp
    });
}

#[bench]
fn bench_f_square(b: &mut test::Bencher) {
    const SAMPLES: usize = 1000;

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    let v: Vec<F> = (0..SAMPLES).map(|_| F::uniform(&mut rng)).collect();

    let mut count = 0;
    b.iter(|| {
        let mut tmp = v[count];
        tmp.square_in_place();
        count = (count + 1) % SAMPLES;
        tmp
    });
}

#[bench]
fn bench_f_inverse(b: &mut test::Bencher) {
    const SAMPLES: usize = 1000;

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    let v: Vec<F> = (0..SAMPLES).map(|_| F::uniform(&mut rng)).collect();

    let mut count = 0;
    b.iter(|| {
        count = (count + 1) % SAMPLES;
        v[count].inverse()
    });
}

#[bench]
fn bench_f_negate(b: &mut test::Bencher) {
    const SAMPLES: usize = 1000;

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    let v: Vec<F> = (0..SAMPLES).map(|_| F::uniform(&mut rng)).collect();

    let mut count = 0;
    b.iter(|| {
        let mut tmp = v[count];
        tmp = tmp.neg();
        count = (count + 1) % SAMPLES;
        tmp
    });
}

#[bench]
fn bench_f_into_repr(b: &mut test::Bencher) {
    const SAMPLES: usize = 1000;

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    let v: Vec<F> = (0..SAMPLES).map(|_| F::uniform(&mut rng)).collect();

    let mut count = 0;
    b.iter(|| {
        count = (count + 1) % SAMPLES;
        v[count].into_repr()
    });
}

#[bench]
fn bench_f_from_repr(b: &mut test::Bencher) {
    const SAMPLES: usize = 1000;

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);

    let v: Vec<BigInteger> = (0..SAMPLES)
        .map(|_| F::uniform(&mut rng).into_repr())
        .collect();

    let mut count = 0;
    b.iter(|| {
        count = (count + 1) % SAMPLES;
        F::from_repr(v[count])
    });
}
