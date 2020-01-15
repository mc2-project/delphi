use crate::*;
use std::{os::raw::c_char, ptr::null_mut, slice::from_raw_parts};

pub struct SEALClientTriples<'a> {
    cfhe:        &'a ClientFHE,
    num_triples: i32,
    a_share:     *const u64,
    b_share:     *const u64,
    a_ct:        *mut c_char,
    b_ct:        *mut c_char,
    c_share:     *mut u64,
}

impl<'a> SEALClientTriples<'a> {
    pub fn new(cfhe: &'a ClientFHE, num_triples: i32, a_vec: &Vec<u64>, b_vec: &Vec<u64>) -> Self {
        Self {
            cfhe,
            num_triples,
            a_share: a_vec.as_ptr(),
            b_share: b_vec.as_ptr(),
            a_ct: null_mut(),
            b_ct: null_mut(),
            c_share: null_mut(),
        }
    }

    pub fn preprocess(&mut self) -> (Vec<c_char>, Vec<c_char>) {
        let mut ct_size: u64 = 0;
        self.a_ct = unsafe {
            triples_preprocess(
                self.a_share,
                self.cfhe.encoder,
                self.cfhe.encryptor,
                self.num_triples,
                &mut ct_size,
            )
        };
        self.b_ct = unsafe {
            triples_preprocess(
                self.b_share,
                self.cfhe.encoder,
                self.cfhe.encryptor,
                self.num_triples,
                &mut ct_size,
            )
        };
        unsafe {
            (
                from_raw_parts(self.a_ct, ct_size as usize).to_vec(),
                from_raw_parts(self.b_ct, ct_size as usize).to_vec(),
            )
        }
    }

    pub fn decrypt(&mut self, mut c_ct: Vec<c_char>) -> Vec<u64> {
        self.c_share = unsafe {
            client_triples_decrypt(
                self.a_share,
                self.b_share,
                c_ct.as_mut_ptr(),
                self.cfhe,
                self.num_triples,
            )
        };
        unsafe { from_raw_parts(self.c_share, self.num_triples as usize).to_vec() }
    }
}

impl<'a> Drop for SEALClientTriples<'a> {
    fn drop(&mut self) {
        unsafe {
            triples_free(self.c_share as *mut i8);
            triples_free(self.a_ct);
            triples_free(self.b_ct);
        }
    }
}
