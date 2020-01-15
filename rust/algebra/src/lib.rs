#![deny(
    unused_import_braces,
    unused_qualifications,
    trivial_casts,
    trivial_numeric_casts
)]
#![deny(unused_qualifications, variant_size_differences, stable_features)]
#![deny(
    non_shorthand_field_patterns,
    unused_attributes,
    unused_imports,
    unused_extern_crates
)]
#![deny(
    renamed_and_removed_lints,
    stable_features,
    unused_allocation,
    unused_comparisons
)]
#![deny(
    unused_must_use,
    unused_mut,
    unused_unsafe,
    private_in_public,
    unsafe_code
)]
#![deny(unsafe_code)]
#![feature(const_fn)]

#[macro_use]
extern crate derivative;

use rand::{CryptoRng, RngCore};

#[cfg_attr(test, macro_use)]
pub mod bytes;
pub use self::bytes::*;

pub mod biginteger;
pub use self::biginteger::*;

pub mod fields;
pub use self::fields::*;

pub mod fixed_point;
pub use fixed_point::*;

pub mod polynomial;
pub use polynomial::*;

pub trait UniformRandom {
    /// Samples a uniformly random field element.
    fn uniform<R: RngCore + CryptoRng>(r: &mut R) -> Self;
}
