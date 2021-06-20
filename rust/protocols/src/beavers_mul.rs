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
    client_gen::{ClientGen, SealClientGen},
    server_gen::{SealServerGen, ServerGen},
    ClientFHE, ServerFHE,
};
use rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use std::{
    io::{Read, Write},
    marker::PhantomData,
    os::raw::c_char,
};

pub struct BeaversMulProtocol<P: FixedPointParameters> {
    index: usize,
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
        let server_gen = SealServerGen::new(&sfhe);

        // Generate shares for a, b, and c
        let mut a = Vec::with_capacity(num_triples);
        let mut b = Vec::with_capacity(num_triples);
        let mut c = Vec::with_capacity(num_triples);
        let mut r = Vec::with_capacity(num_triples);
        for i in 0..num_triples {
            a.push(P::Field::uniform(rng));
            b.push(P::Field::uniform(rng));
            c.push(P::Field::uniform(rng));
            r.push(a[i] * b[i] - c[i]);
        }

        let mut server_triples = server_gen.triples_preprocess(
            a.iter()
                .map(|e| e.into_repr().0)
                .collect::<Vec<_>>()
                .as_slice(),
            b.iter()
                .map(|e| e.into_repr().0)
                .collect::<Vec<_>>()
                .as_slice(),
            r.iter()
                .map(|e| e.into_repr().0)
                .collect::<Vec<_>>()
                .as_slice(),
        );

        // Receive clients Enc(a1), Enc(b1)
        let recv_message: OfflineMsgRcv = crate::bytes::deserialize(reader)?;
        let recv_struct = recv_message.msg();
        let mut a_ct = recv_struct.a_shares;
        let mut b_ct = recv_struct.b_shares;

        // Compute and send Enc(a1b2 + b1a2 + a2b2 - r) to client
        let c_ct = server_gen.triples_online(
            &mut server_triples,
            a_ct.as_mut_slice(),
            b_ct.as_mut_slice(),
        );
        let client_msg = BeaversOfflineMsg {
            a_shares: c_ct,
            b_shares: Vec::new(),
        };
        let send_message = OfflineMsgSend::new(&client_msg);
        crate::bytes::serialize(writer, &send_message)?;

        let mut server_triples: Vec<Triple<P>> = Vec::with_capacity(num_triples);
        for idx in 0..num_triples {
            server_triples.push(crypto_primitives::Triple {
                a: a[idx],
                b: b[idx],
                c: c[idx],
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
        let client_gen = SealClientGen::new(&cfhe);

        // Generate shares for a and b
        let mut a = Vec::with_capacity(num_triples);
        let mut b = Vec::with_capacity(num_triples);
        for _ in 0..num_triples {
            a.push(P::Field::uniform(rng));
            b.push(P::Field::uniform(rng));
        }
        // Compute Enc(a1), Enc(b1)
        let (mut client_triples, a_ct, b_ct) = client_gen.triples_preprocess(
            a.iter()
                .map(|e| e.into_repr().0)
                .collect::<Vec<_>>()
                .as_slice(),
            b.iter()
                .map(|e| e.into_repr().0)
                .collect::<Vec<_>>()
                .as_slice(),
        );

        // Send Enc(a1), Enc(b1) to server
        let server_message = BeaversOfflineMsg {
            a_shares: a_ct,
            b_shares: b_ct,
        };
        let send_message = OfflineMsgSend::new(&server_message);
        crate::bytes::serialize(writer, &send_message)?;

        // Receive Enc(a1b2 + b1a2 + a2b2 - r) from server
        let recv_message: OfflineMsgRcv = crate::bytes::deserialize(reader)?;
        let mut recv_struct = recv_message.msg();

        // Compute and decrypt Enc(ab - r) and construct client triples
        let c_share_vec = client_gen
            .triples_postprocess(&mut client_triples, recv_struct.a_shares.as_mut_slice());

        let mut client_triples: Vec<Triple<P>> = Vec::with_capacity(num_triples);
        for idx in 0..num_triples {
            client_triples.push(crypto_primitives::Triple {
                a: a[idx],
                b: b[idx],
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
