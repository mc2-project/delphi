use crate::*;
use std::os::raw::c_char;

pub trait ServerGen {
    type Keys;
    /// The type of messages passed between client and server
    type MsgType;

    /// Create new ServerGen object
    fn new(keys: Self::Keys) -> Self;

    /// Preprocess `a`, `b`, and `c` randomizers
    fn triples_preprocess(&self, a: &[u64], b: &[u64], r: &[u64]) -> ServerTriples;

    /// Process clients's input and return `c` shares for client
    fn triples_online(
        &self,
        shares: &mut ServerTriples,
        a: &mut [Self::MsgType],
        b: &mut [Self::MsgType],
    ) -> Vec<Self::MsgType>;
}

/// SEAL implementation of ClientGen
pub struct SealServerGen<'a> {
    sfhe: &'a ServerFHE,
}

impl<'a> ServerGen for SealServerGen<'a> {
    type Keys = &'a ServerFHE;
    /// Messages are SEAL ciphertexts which are passed as opaque C pointers
    type MsgType = c_char;

    fn new(sfhe: Self::Keys) -> Self {
        Self { sfhe }
    }

    fn triples_preprocess(&self, a: &[u64], b: &[u64], r: &[u64]) -> ServerTriples {
        unsafe {
            server_triples_preprocess(
                self.sfhe,
                a.len() as u32,
                a.as_ptr(),
                b.as_ptr(),
                r.as_ptr(),
            )
        }
    }

    fn triples_online(
        &self,
        shares: &mut ServerTriples,
        a: &mut [Self::MsgType],
        b: &mut [Self::MsgType],
    ) -> Vec<Self::MsgType> {
        let a_ct = SerialCT {
            inner: a.as_mut_ptr(),
            size: a.len() as u64,
        };
        let b_ct = SerialCT {
            inner: b.as_mut_ptr(),
            size: b.len() as u64,
        };
        unsafe { server_triples_online(self.sfhe, a_ct, b_ct, shares) };
        let result = unsafe {
            std::slice::from_raw_parts(shares.c_ct.inner, shares.c_ct.size as usize).to_vec()
        };
        result
    }
}
