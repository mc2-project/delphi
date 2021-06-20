macro_rules! impl_field_into_bigint {
    ($field: ident, $bigint: ident, $params: ident) => {
        impl<P: $params> From<$field<P>> for $bigint {
            fn from(val: $field<P>) -> Self {
                val.into_repr()
            }
        }
    };
}

macro_rules! impl_ops_traits {
    ($field: ident, $params: ident) => {
        impl<P: $params> Add<Self> for $field<P> {
            type Output = $field<P>;

            #[inline]
            fn add(self, other: Self) -> Self {
                self + &other
            }
        }

        impl<P: $params> Sub<Self> for $field<P> {
            type Output = $field<P>;

            #[inline]
            fn sub(self, other: Self) -> Self {
                self - &other
            }
        }

        impl<P: $params> Mul<Self> for $field<P> {
            type Output = $field<P>;

            #[inline]
            fn mul(self, other: Self) -> Self {
                self * &other
            }
        }

        impl<P: $params> Div<Self> for $field<P> {
            type Output = $field<P>;

            #[inline]
            fn div(self, other: Self) -> Self {
                self / &other
            }
        }

        impl<P: $params> AddAssign<Self> for $field<P> {
            #[inline]
            fn add_assign(&mut self, other: Self) {
                *self += &other
            }
        }

        impl<P: $params> SubAssign<Self> for $field<P> {
            #[inline]
            fn sub_assign(&mut self, other: Self) {
                *self -= &other
            }
        }

        impl<P: $params> MulAssign<Self> for $field<P> {
            #[inline]
            fn mul_assign(&mut self, other: Self) {
                *self *= &other
            }
        }

        impl<P: $params> DivAssign<Self> for $field<P> {
            #[inline]
            fn div_assign(&mut self, other: Self) {
                *self /= &other
            }
        }
    };
}

macro_rules! sqrt_impl {
    ($Self:ident, $P:tt, $self:expr) => {{
        use crate::fields::LegendreSymbol::*;
        // https://eprint.iacr.org/2012/685.pdf (page 12, algorithm 5)
        // Actually this is just normal Tonelli-Shanks; since `P::Generator`
        // is a quadratic non-residue, `P::ROOT_OF_UNITY = P::GENERATOR ^ t`
        // is also a quadratic non-residue (since `t` is odd).
        match $self.legendre() {
            Zero => Some(*$self),
            QuadraticNonResidue => None,
            QuadraticResidue => {
                let mut z = $Self::qnr_to_t();
                let mut w = $self.pow($P::T_MINUS_ONE_DIV_TWO);
                let mut x = w * $self;
                let mut b = x * &w;

                let mut v = $P::TWO_ADICITY as usize;
                // t = self^t
                #[cfg(debug_assertions)]
                {
                    let mut check = b;
                    for _ in 0..(v - 1) {
                        check.square_in_place();
                    }
                    if !check.is_one() {
                        panic!("Input is not a square root, but it passed the QR test")
                    }
                }

                while !b.is_one() {
                    let mut k = 0usize;

                    let mut b2k = b;
                    while !b2k.is_one() {
                        // invariant: b2k = b^(2^k) after entering this loop
                        b2k.square_in_place();
                        k += 1;
                    }

                    let j = v - k - 1;
                    w = z;
                    for _ in 0..j {
                        w.square_in_place();
                    }

                    z = w.square();
                    b *= &z;
                    x *= &w;
                    v = k;
                }

                Some(x)
            }
        }
    }};
}
