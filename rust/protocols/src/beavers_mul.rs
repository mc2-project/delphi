#![allow(dead_code)]
use crate::{AdditiveShare, InMessage, OutMessage};
use algebra::{
    fields::PrimeField,
    fixed_point::{FixedPoint, FixedPointParameters},
    fp_64::Fp64Parameters,
    FpParameters, UniformRandom,
};
use crypto_primitives::{BeaversMul, BlindedSharedInputs};
use io_utils::IMuxSync;
use protocols_sys::{
    client_triples::SEALClientTriples, server_triples::SEALServerTriples, ClientFHE, ServerFHE,
};
use rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use std::{
    io::{Read, Write},
    marker::PhantomData,
    os::raw::c_char,
};

pub struct BeaversMulProtocol<P: FixedPointParameters> {
    index:  usize,
    _share: PhantomData<P>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BeaversOfflineMsg {
    pub a_shares: Vec<c_char>,
    pub b_shares: Vec<c_char>,
}

type Triple<P> = crypto_primitives::Triple<<P as FixedPointParameters>::Field>;

pub struct BeaversMulProtocolType;

pub type OfflineMsgSend<'a> = OutMessage<'a, BeaversOfflineMsg, BeaversMulProtocolType>;
pub type OfflineMsgRcv = InMessage<BeaversOfflineMsg, BeaversMulProtocolType>;

pub type OnlineMsgSend<'a, P> =
    OutMessage<'a, [BlindedSharedInputs<FixedPoint<P>>], BeaversMulProtocolType>;
pub type OnlineMsgRcv<P> =
    InMessage<Vec<BlindedSharedInputs<FixedPoint<P>>>, BeaversMulProtocolType>;

impl<P: FixedPointParameters> BeaversMulProtocol<P>
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    pub(crate) fn offline_server_protocol<M, R, W, RNG>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        sfhe: &ServerFHE,
        num_triples: usize,
        rng: &mut RNG,
    ) -> Result<Vec<Triple<P>>, bincode::Error>
    where
        M: BeaversMul<FixedPoint<P>>,
        R: Read + Send,
        W: Write + Send,
        RNG: RngCore + CryptoRng,
    {
        // Generate shares for a, b, and c
        let mut a: Vec<FixedPoint<P>> = Vec::with_capacity(num_triples);
        let mut b: Vec<FixedPoint<P>> = Vec::with_capacity(num_triples);
        let mut r: Vec<FixedPoint<P>> = Vec::with_capacity(num_triples);
        for _ in 0..num_triples {
            a.push(FixedPoint::new(P::Field::uniform(rng)));
            b.push(FixedPoint::new(P::Field::uniform(rng)));
            r.push(FixedPoint::new(P::Field::uniform(rng)));
        }
        // Convert a, b, r to u64 Vec
        let a_c: Vec<u64> = a.iter().map(|e| e.inner.into_repr().0).collect();
        let b_c: Vec<u64> = b.iter().map(|e| e.inner.into_repr().0).collect();
        let r_c: Vec<u64> = r.iter().map(|e| e.inner.into_repr().0).collect();

        let mut seal_server = SEALServerTriples::new(sfhe, num_triples as i32, &a_c, &b_c, &r_c);

        // Receive clients Enc(a1), Enc(b1)
        let recv_message: OfflineMsgRcv = crate::bytes::deserialize(reader)?;
        let recv_struct = recv_message.msg();
        let a_ct = recv_struct.a_shares;
        let b_ct = recv_struct.b_shares;

        // Compute and send Enc(a1b2 + b1a2 + a2b2 - r) to client
        let client_share_ct_vec = seal_server.process(a_ct, b_ct);
        let client_msg = BeaversOfflineMsg {
            a_shares: client_share_ct_vec,
            b_shares: Vec::new(),
        };
        let send_message = OfflineMsgSend::new(&client_msg);
        crate::bytes::serialize(writer, &send_message)?;

        let mut server_triples: Vec<Triple<P>> = Vec::with_capacity(num_triples);
        for idx in 0..num_triples {
            server_triples.push(crypto_primitives::Triple {
                a: a[idx].inner,
                b: b[idx].inner,
                c: r[idx].inner,
            });
        }
        Ok(server_triples)
    }

    pub(crate) fn offline_client_protocol<
        M: BeaversMul<FixedPoint<P>>,
        R: Read + Send,
        W: Write + Send,
        RNG: RngCore + CryptoRng,
    >(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        cfhe: &ClientFHE,
        num_triples: usize,
        rng: &mut RNG,
    ) -> Result<Vec<Triple<P>>, bincode::Error> {
        // Generate shares for a and b
        let mut a: Vec<FixedPoint<P>> = Vec::with_capacity(num_triples);
        let mut b: Vec<FixedPoint<P>> = Vec::with_capacity(num_triples);
        for _ in 0..num_triples {
            a.push(FixedPoint::new(P::Field::uniform(rng)));
            b.push(FixedPoint::new(P::Field::uniform(rng)));
        }
        // Compute Enc(a1), Enc(b1)
        let a_c: Vec<u64> = a.iter().map(|e| e.inner.into_repr().0).collect();
        let b_c: Vec<u64> = b.iter().map(|e| e.inner.into_repr().0).collect();

        let mut seal_client = SEALClientTriples::new(cfhe, num_triples as i32, &a_c, &b_c);
        let (a_ct_vec, b_ct_vec) = seal_client.preprocess();

        // Send Enc(a1), Enc(b1) to server
        let server_message = BeaversOfflineMsg {
            a_shares: a_ct_vec,
            b_shares: b_ct_vec,
        };
        let send_message = OfflineMsgSend::new(&server_message);
        crate::bytes::serialize(writer, &send_message)?;

        // Receive Enc(a1b2 + b1a2 + a2b2 - r) from server
        let recv_message: OfflineMsgRcv = crate::bytes::deserialize(reader)?;
        let recv_struct = recv_message.msg();

        // Compute and decrypt Enc(ab - r) and construct client triples
        let c_share_vec = seal_client.decrypt(recv_struct.a_shares);

        let mut client_triples: Vec<Triple<P>> = Vec::with_capacity(num_triples);
        for idx in 0..num_triples {
            client_triples.push(crypto_primitives::Triple {
                a: a[idx].inner,
                b: b[idx].inner,
                c: P::Field::from_repr(c_share_vec[idx].into()),
            });
        }
        Ok(client_triples)
    }

    pub fn online_server_protocol<M: BeaversMul<FixedPoint<P>>, R: Read + Send, W: Write + Send>(
        party_index: usize,
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        x_s: &[AdditiveShare<P>],
        y_s: &[AdditiveShare<P>],
        triples: &[Triple<P>],
    ) -> Result<Vec<AdditiveShare<P>>, bincode::Error> {
        // Compute blinded shares using the triples.
        let self_blinded_and_shared = x_s
            .iter()
            .zip(y_s)
            .zip(triples.iter())
            .map(|((x, y), triple)| M::share_and_blind_inputs(x, y, &triple))
            .collect::<Vec<_>>();

        let mut result = Vec::with_capacity(triples.len());
        rayon::scope(|s| {
            s.spawn(|_| {
                for msg_contents in self_blinded_and_shared.chunks(8192) {
                    let sent_message = OnlineMsgSend::new(&msg_contents);
                    crate::bytes::serialize(writer, &sent_message).unwrap();
                }
            });
            s.spawn(|_| {
                let num_chunks = (triples.len() as f64 / 8192.0).ceil() as usize;
                for _ in 0..num_chunks {
                    let in_msg: OnlineMsgRcv<_> = crate::bytes::deserialize(reader).unwrap();
                    let shares = in_msg.msg();
                    result.extend(shares);
                }
            });
        });

        // TODO: use rayon to spawn this on a different thread.
        Ok(result
            .into_iter()
            .zip(self_blinded_and_shared)
            .map(|(cur, other)| M::reconstruct_blinded_inputs(cur, other))
            .zip(triples)
            .map(|(inp, triple)| M::multiply_blinded_inputs(party_index, inp, triple))
            .collect())
    }

    pub fn online_client_protocol<M: BeaversMul<FixedPoint<P>>, R: Read + Send, W: Write + Send>(
        party_index: usize,
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        x_s: &[AdditiveShare<P>],
        y_s: &[AdditiveShare<P>],
        triples: &[Triple<P>],
    ) -> Result<Vec<AdditiveShare<P>>, bincode::Error> {
        // Compute blinded shares using the triples.
        let self_blinded_and_shared = x_s
            .iter()
            .zip(y_s)
            .zip(triples.iter())
            .map(|((x, y), triple)| M::share_and_blind_inputs(x, y, &triple))
            .collect::<Vec<_>>();

        let mut result = Vec::with_capacity(triples.len());
        rayon::scope(|s| {
            s.spawn(|_| {
                let num_chunks = (triples.len() as f64 / 8192.0).ceil() as usize;
                for _ in 0..num_chunks {
                    let in_msg: OnlineMsgRcv<_> = crate::bytes::deserialize(reader).unwrap();
                    let shares = in_msg.msg();
                    result.extend(shares);
                }
            });
            s.spawn(|_| {
                // TODO: use rayon to spawn this on a different thread.
                for msg_contents in self_blinded_and_shared.chunks(8192) {
                    let sent_message = OnlineMsgSend::new(&msg_contents);
                    crate::bytes::serialize(writer, &sent_message).unwrap();
                }
            });
        });

        // TODO: use rayon to spawn this on a different thread.
        Ok(result
            .into_iter()
            .zip(self_blinded_and_shared)
            .map(|(cur, other)| M::reconstruct_blinded_inputs(cur, other))
            .zip(triples)
            .map(|(inp, triple)| M::multiply_blinded_inputs(party_index, inp, triple))
            .collect())
    }
}
