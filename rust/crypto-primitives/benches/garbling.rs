use criterion::{criterion_group, criterion_main, Bencher, Criterion};

use std::time::Duration;

use fancy_garbling::{
    circuit::{Circuit, CircuitBuilder},
    util::RngExt,
};

use algebra::{fields::near_mersenne_64::F, *};

struct TenBitExpParams {}
impl FixedPointParameters for TenBitExpParams {
    type Field = F;
    const MANTISSA_CAPACITY: u8 = 3;
    const EXPONENT_CAPACITY: u8 = 10;
}

fn make_relu(n: usize) -> Circuit {
    let mut b = CircuitBuilder::new();
    crypto_primitives::gc::relu::<TenBitExpParams>(&mut b, n).unwrap();
    b.finish()
}

fn relu_gb(c: &mut Criterion) {
    c.bench_function_over_inputs(
        &"relu_gb",
        move |bench: &mut Bencher, &num: &&usize| {
            let mut c = make_relu(*num);
            bench.iter(|| {
                let gb = fancy_garbling::garble(&mut c).unwrap();
                criterion::black_box(gb);
            });
        },
        &[1, 10, 100usize],
    );
}

fn relu_ev(c: &mut Criterion) {
    c.bench_function_over_inputs(
        &"relu_ev",
        move |bench: &mut Bencher, &num: &&usize| {
            let mut rng = rand::thread_rng();
            let mut c = make_relu(*num);
            let (en, ev) = fancy_garbling::garble(&mut c).unwrap();
            let gb_inps: Vec<_> = (0..c.num_garbler_inputs())
                .map(|i| rng.gen_u16() % c.garbler_input_mod(i))
                .collect();
            let ev_inps: Vec<_> = (0..c.num_evaluator_inputs())
                .map(|i| rng.gen_u16() % c.evaluator_input_mod(i))
                .collect();
            let xs = en.encode_garbler_inputs(&gb_inps);
            let ys = en.encode_evaluator_inputs(&ev_inps);
            bench.iter(|| {
                let ys = ev.eval(&mut c, &xs, &ys).unwrap();
                criterion::black_box(ys);
            });
        },
        &[1, 10, 100usize],
    );
}

criterion_group! {
    name = garbling;
    config = Criterion::default().warm_up_time(Duration::from_millis(100)).sample_size(10);
    targets = relu_gb, relu_ev,
}

criterion_main!(garbling);
