use crate::*;
use std::{os::raw::c_char, ptr::null_mut, slice::from_raw_parts};

pub struct SEALServerTriples<'a> {
    sfhe:         &'a ServerFHE,
    num_triples:  i32,
    a_share:      *const u64,
    b_share:      *const u64,
    c_share:      *const u64,
    client_share: *mut c_char,
}

impl<'a> SEALServerTriples<'a> {
    pub fn new(
        sfhe: &'a ServerFHE,
        num_triples: i32,
        a_vec: &Vec<u64>,
        b_vec: &Vec<u64>,
        c_vec: &Vec<u64>,
    ) -> Self {
        Self {
            sfhe,
            num_triples,
            a_share: a_vec.as_ptr(),
            b_share: b_vec.as_ptr(),
            c_share: c_vec.as_ptr(),
            client_share: null_mut(),
        }
    }

    pub fn process(&mut self, mut a_ct: Vec<c_char>, mut b_ct: Vec<c_char>) -> Vec<c_char> {
        let mut enc_size: u64 = 0;
        self.client_share = unsafe {
            server_triples_online(
                a_ct.as_mut_ptr(),
                b_ct.as_mut_ptr(),
                self.a_share,
                self.b_share,
                self.c_share,
                self.sfhe,
                self.num_triples,
                &mut enc_size,
            )
        };
        unsafe { from_raw_parts(self.client_share, enc_size as usize).to_vec() }
    }
}

impl<'a> Drop for SEALServerTriples<'a> {
    fn drop(&mut self) {
        unsafe { triples_free(self.client_share) }
    }
}
