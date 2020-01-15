use algebra::fixed_point::FixedPoint;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[macro_use]
extern crate bench_utils;

extern crate ndarray;

pub mod beavers_mul;
pub mod gc;
pub mod linear_layer;
pub mod neural_network;
pub mod quad_approx;

mod bytes;

#[cfg(test)]
mod tests;

pub type AdditiveShare<P> = crypto_primitives::AdditiveShare<FixedPoint<P>>;

#[derive(Serialize)]
pub struct OutMessage<'a, T: 'a + ?Sized, Type> {
    msg:           &'a T,
    protocol_type: PhantomData<Type>,
}

impl<'a, T: 'a + ?Sized, Type> OutMessage<'a, T, Type> {
    pub fn new(msg: &'a T) -> Self {
        Self {
            msg,
            protocol_type: PhantomData,
        }
    }

    pub fn msg(&self) -> &T {
        self.msg
    }
}

#[derive(Deserialize)]
pub struct InMessage<T, Type> {
    msg:           T,
    protocol_type: PhantomData<Type>,
}

impl<T, Type> InMessage<T, Type> {
    pub fn msg(self) -> T {
        self.msg
    }
}
