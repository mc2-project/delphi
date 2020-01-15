macro_rules! ndarray_impl {
    ($name:ident, $inner:ident, $dims:ty) => {
        #[derive(Clone, PartialEq, Eq, Debug, Default, Hash, Serialize, Deserialize)]
        pub struct $name<F>($inner<F>);

        impl<F> $name<F> {
            #[inline]
            pub fn from_elem(dim: $dims, elem: F) -> Self
            where
                F: Clone,
            {
                $name($inner::from_elem(dim, elem))
            }

            #[inline]
            pub fn zeros(dim: $dims) -> Self
            where
                F: Zero + Clone,
            {
                $name($inner::zeros(dim))
            }

            #[inline]
            pub fn from_shape_vec(dim: $dims, v: Vec<F>) -> Result<Self, ndarray::ShapeError> {
                $inner::from_shape_vec(dim, v).map($name)
            }

            #[inline]
            pub fn iter(&self) -> impl Iterator<Item = &F> {
                self.0.iter()
            }

            #[inline]
            pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut F> {
                self.0.iter_mut()
            }
        }

        impl<T: From<I>, I: Copy> From<$inner<I>> for $name<T> {
            #[inline]
            fn from(other: $inner<I>) -> Self {
                let inp_vec: Vec<T> = other.iter().map(|e| T::from(*e)).collect();
                $name(
                    ndarray::Array1::from_vec(inp_vec)
                        .into_shape(other.dim())
                        .unwrap(),
                )
            }
        }

        impl<F> std::ops::Deref for $name<F> {
            type Target = $inner<F>;

            #[inline]
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<F> std::ops::DerefMut for $name<F> {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl<'a, F> IntoIterator for &'a $name<F> {
            type Item = &'a F;
            type IntoIter = <&'a $inner<F> as IntoIterator>::IntoIter;

            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                self.0.into_iter()
            }
        }

        impl<'a, F> IntoIterator for &'a mut $name<F> {
            type Item = &'a mut F;
            type IntoIter = <&'a mut $inner<F> as IntoIterator>::IntoIter;

            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                (&mut self.0).into_iter()
            }
        }
    };
}
