use crate::{beavers_mul::BeaversMulProtocol, AdditiveShare};
use algebra::{
    fields::PrimeField,
    fixed_point::{FixedPoint, FixedPointParameters},
    fp_64::Fp64Parameters,
    FpParameters, Polynomial,
};
use crypto_primitives::{BeaversMul, Triple};
use io_utils::IMuxSync;
use protocols_sys::*;
use rand::{CryptoRng, RngCore};
use std::{
    io::{Read, Write},
    marker::PhantomData,
};

pub struct QuadApproxProtocol<P: FixedPointParameters> {
    _share: PhantomData<P>,
}

impl<P: FixedPointParameters> QuadApproxProtocol<P>
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    pub fn offline_server_protocol<
        M: BeaversMul<FixedPoint<P>>,
        R: Read + Send,
        W: Write + Send,
        RNG: RngCore + CryptoRng,
    >(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        sfhe: &ServerFHE,
        num_approx: usize,
        rng: &mut RNG,
    ) -> Result<Vec<Triple<P::Field>>, bincode::Error> {
        if num_approx != 0 {
            BeaversMulProtocol::offline_server_protocol::<M, _, _, _>(
                reader, writer, sfhe, num_approx, rng,
            )
        } else {
            Ok(Vec::new())
        }
    }

    pub fn offline_client_protocol<
        M: BeaversMul<FixedPoint<P>>,
        R: Read + Send,
        W: Write + Send,
        RNG: RngCore + CryptoRng,
    >(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        cfhe: &ClientFHE,
        num_approx: usize,
        rng: &mut RNG,
    ) -> Result<Vec<Triple<P::Field>>, bincode::Error> {
        if num_approx != 0 {
            BeaversMulProtocol::offline_client_protocol::<M, _, _, _>(
                reader, writer, cfhe, num_approx, rng,
            )
        } else {
            Ok(Vec::new())
        }
    }

    pub fn online_server_protocol<M: BeaversMul<FixedPoint<P>>, R: Read + Send, W: Write + Send>(
        party_index: usize,
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        polynomial: &Polynomial<FixedPoint<P>>,
        x_s: &[AdditiveShare<P>],
        triples: &[Triple<P::Field>],
    ) -> Result<Vec<AdditiveShare<P>>, bincode::Error> {
        let mut x_squared = BeaversMulProtocol::online_server_protocol::<M, R, W>(
            party_index,
            reader,
            writer,
            x_s,
            x_s,
            &triples,
        )?;
        let coeffs = polynomial.coeffs();
        assert_eq!(coeffs.len(), 3);
        let a_0 = coeffs[0];
        let a_1 = coeffs[1];
        let a_2 = coeffs[2];

        // Reduce down to correct size.
        for (x2, x) in x_squared.iter_mut().zip(x_s) {
            x2.inner.signed_reduce_in_place();
            *x2 *= a_2;
            x2.inner.signed_reduce_in_place();
            let mut a_1_x = *x * a_1;
            a_1_x.inner.signed_reduce_in_place();
            *x2 += a_1_x;
            // Only add the constant if we are the client.
            // (Both parties adding it would mean that the constant is doubled.)
            if party_index == crate::neural_network::CLIENT {
                x2.add_constant_in_place(a_0);
            }
        }
        Ok(x_squared)
    }

    pub fn online_client_protocol<M: BeaversMul<FixedPoint<P>>, R: Read + Send, W: Write + Send>(
        party_index: usize,
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        polynomial: &Polynomial<FixedPoint<P>>,
        x_s: &[AdditiveShare<P>],
        triples: &[Triple<P::Field>],
    ) -> Result<Vec<AdditiveShare<P>>, bincode::Error> {
        let mut x_squared = BeaversMulProtocol::online_client_protocol::<M, R, W>(
            party_index,
            reader,
            writer,
            x_s,
            x_s,
            &triples,
        )?;
        let coeffs = polynomial.coeffs();
        assert_eq!(coeffs.len(), 3);
        let a_0 = coeffs[0];
        let a_1 = coeffs[1];
        let a_2 = coeffs[2];

        // Reduce down to correct size.
        for (x2, x) in x_squared.iter_mut().zip(x_s) {
            x2.inner.signed_reduce_in_place();
            *x2 *= a_2;
            x2.inner.signed_reduce_in_place();
            let mut a_1_x = *x * a_1;
            a_1_x.inner.signed_reduce_in_place();
            *x2 += a_1_x;
            // Only add the constant if we are the client.
            // (Both parties adding it would mean that the constant is doubled.)
            if party_index == crate::neural_network::CLIENT {
                x2.add_constant_in_place(a_0);
            }
        }
        Ok(x_squared)
    }
}
