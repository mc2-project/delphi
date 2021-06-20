#![allow(
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code
)]

#[macro_use]
pub extern crate ndarray;

pub mod client_cg;
pub mod client_gen;
pub mod key_share;
pub mod server_cg;
pub mod server_gen;

pub use client_cg::*;
pub use client_gen::*;
pub use key_share::KeyShare;
pub use server_cg::*;
pub use server_gen::*;

use std::os::raw::c_char;

include!("bindings.rs");

pub struct SealCT {
    pub inner: SerialCT,
}

impl SealCT {
    pub fn new() -> Self {
        let inner = SerialCT {
            inner: std::ptr::null_mut(),
            size: 0,
        };
        Self { inner }
    }

    /// Encrypt a vector using SEAL
    pub fn encrypt_vec(&mut self, cfhe: &ClientFHE, input: Vec<u64>) -> Vec<c_char> {
        self.inner = unsafe { encrypt_vec(cfhe, input.as_ptr(), input.len() as u64) };

        unsafe { std::slice::from_raw_parts(self.inner.inner, self.inner.size as usize).to_vec() }
    }

    /// Decrypt a vector of SEAL ciphertexts. Assumes `inner.share_size` is set.
    pub fn decrypt_vec(&mut self, cfhe: &ClientFHE, mut ct: Vec<c_char>, size: usize) -> Vec<u64> {
        // Don't replace the current inner CT, since the received ciphertext was
        // allocated by Rust
        let mut recv_ct = SerialCT {
            inner: ct.as_mut_ptr(),
            size: ct.len() as u64,
        };
        unsafe {
            let raw_vec = decrypt_vec(cfhe, &mut recv_ct, size as u64);
            std::slice::from_raw_parts(raw_vec, size as usize).to_vec()
        }
    }
}

impl Drop for ClientFHE {
    fn drop(&mut self) {
        unsafe {
            client_free_keys(self);
        }
    }
}
unsafe impl Send for ClientFHE {}
unsafe impl Sync for ClientFHE {}

impl Drop for ServerFHE {
    fn drop(&mut self) {
        unsafe {
            server_free_keys(self);
        }
    }
}
unsafe impl Send for ServerFHE {}
unsafe impl Sync for ServerFHE {}

impl Drop for ClientTriples {
    fn drop(&mut self) {
        unsafe {
            client_triples_free(self);
        }
    }
}

unsafe impl Send for ClientTriples {}
unsafe impl Sync for ClientTriples {}

impl Drop for ServerTriples {
    fn drop(&mut self) {
        unsafe {
            server_triples_free(self);
        }
    }
}

unsafe impl Send for ServerTriples {}
unsafe impl Sync for ServerTriples {}
