//! Some core cryptographic primitives used by Delphi.
#![deny(unused_import_braces, unused_qualifications, trivial_casts)]
#![deny(trivial_numeric_casts, private_in_public, variant_size_differences)]
#![deny(stable_features, unreachable_pub, non_shorthand_field_patterns)]
#![deny(unused_attributes, unused_imports, unused_mut, missing_docs)]
#![deny(renamed_and_removed_lints, stable_features, unused_allocation)]
#![deny(unused_comparisons, bare_trait_objects, unused_must_use)]
#![forbid(unsafe_code)]

/// Defines `struct`s and `trait`s for constructing additively-shared ring
/// elements.
pub mod additive_share;
/// Defines `struct`s and `trait`s for multiplying additive shares.
pub mod beavers_mul;
/// Generates a circuit for computing the ReLU of an additively shared value.
pub mod gc;

pub use additive_share::{AdditiveShare, Share};
pub use beavers_mul::*;
pub use gc::relu;
