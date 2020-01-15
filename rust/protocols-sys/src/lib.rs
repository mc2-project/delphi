#![allow(
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code
)]

#[macro_use]
pub extern crate ndarray;

pub mod client_linear;
pub mod client_triples;
pub mod key_share;
pub mod server_linear;
pub mod server_triples;

include!("bindings.rs");

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
