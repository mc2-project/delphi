use crate::*;
use std::os::raw::c_char;

pub trait ClientGen {
    type Keys;
    /// The type of messages passed between client and server
    type MsgType;

    /// Create new ClientGen object
    fn new(keys: Self::Keys) -> Self;

    /// Preprocess `a` and `b` randomizers for sending to the server
    fn triples_preprocess(
        &self,
        a: &[u64],
        b: &[u64],
    ) -> (ClientTriples, Vec<Self::MsgType>, Vec<Self::MsgType>);

    /// Postprocess server's response and return `c` shares
    fn triples_postprocess(
        &self,
        shares: &mut ClientTriples,
        c_ct: &mut [Self::MsgType],
    ) -> Vec<u64>;
}

/// SEAL implementation of ClientGen
pub struct SealClientGen<'a> {
    cfhe: &'a ClientFHE,
}

impl<'a> ClientGen for SealClientGen<'a> {
    type Keys = &'a ClientFHE;
    /// Messages are SEAL ciphertexts which are passed as opaque C pointers
    type MsgType = c_char;

    fn new(cfhe: Self::Keys) -> Self {
        Self { cfhe }
    }

    fn triples_preprocess(
        &self,
        a: &[u64],
        b: &[u64],
    ) -> (ClientTriples, Vec<c_char>, Vec<c_char>) {
        let shares =
            unsafe { client_triples_preprocess(self.cfhe, a.len() as u32, a.as_ptr(), b.as_ptr()) };
        let a_ct = shares.a_ct.clone();
        let b_ct = shares.b_ct.clone();
        unsafe {
            (
                shares,
                std::slice::from_raw_parts(a_ct.inner, a_ct.size as usize).to_vec(),
                std::slice::from_raw_parts(b_ct.inner, b_ct.size as usize).to_vec(),
            )
        }
    }

    fn triples_postprocess(&self, shares: &mut ClientTriples, c: &mut [Self::MsgType]) -> Vec<u64> {
        let c_ct = SerialCT {
            inner: c.as_mut_ptr(),
            size: c.len() as u64,
        };
        unsafe {
            client_triples_decrypt(self.cfhe, c_ct, shares);
        };
        let result =
            unsafe { std::slice::from_raw_parts(shares.c_share, shares.num as usize).to_vec() };
        result
    }
}
