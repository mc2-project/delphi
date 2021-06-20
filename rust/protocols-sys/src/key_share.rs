use crate::*;
use std::slice::from_raw_parts;

pub struct KeyShare(SerialCT);

impl KeyShare {
    pub fn new() -> Self {
        Self(SerialCT {
            inner: ::std::ptr::null_mut(),
            size: 0,
        })
    }

    pub fn generate(&mut self) -> (ClientFHE, Vec<std::os::raw::c_char>) {
        let cfhe = unsafe { client_keygen(&mut self.0) };
        (cfhe, unsafe {
            from_raw_parts(self.0.inner, self.0.size as usize).to_vec()
        })
    }

    pub fn receive(&mut self, mut keys_vec: Vec<std::os::raw::c_char>) -> ServerFHE {
        let serial_keys = SerialCT {
            inner: keys_vec.as_mut_ptr(),
            size: keys_vec.len() as u64,
        };
        unsafe { server_keygen(serial_keys) }
    }
}

impl Drop for KeyShare {
    fn drop(&mut self) {
        unsafe {
            free_ct(&mut self.0);
        }
    }
}
