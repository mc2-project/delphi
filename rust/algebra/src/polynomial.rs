use num_traits::{One, Zero};
use std::ops::{AddAssign, Mul, MulAssign};

/// A polynomial with coefficients in `F`.
#[derive(Debug, Clone)]
pub struct Polynomial<F> {
    coeffs: Vec<F>,
}

impl<C> Polynomial<C> {
    /// Constructs a new polynomial p(x) = a_0 + a_1 * x + ... + a_n x^n
    /// when given as inputs the coefficients `coeffs`
    /// such that `coeffs[i] = a_i`.
    pub fn new(coeffs: Vec<C>) -> Self {
        Self { coeffs }
    }

    pub fn evaluate<F>(&self, point: F) -> F
    where
        F: AddAssign + MulAssign + Mul<C, Output = F> + Zero + One + Copy,
        C: Copy,
    {
        let mut sum = F::zero();
        let mut power = F::one();
        for coeff in &self.coeffs {
            sum += power * (*coeff);
            power *= point;
        }
        sum
    }

    pub fn coeffs(&self) -> &[C] {
        &self.coeffs
    }
}
