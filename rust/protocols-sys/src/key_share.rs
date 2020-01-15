use crate::*;
use std::slice::from_raw_parts;

pub struct KeyShare(*mut ::std::os::raw::c_char);

impl KeyShare {
    pub fn new() -> Self {
        Self {
            0: ::std::ptr::null_mut(),
        }
    }

    pub fn generate(&mut self) -> (ClientFHE, Vec<std::os::raw::c_char>) {
        let mut keys_len: u64 = 0;
        let cfhe = unsafe { client_keygen(&mut self.0, &mut keys_len) };
        (cfhe, unsafe {
            from_raw_parts(self.0, keys_len as usize).to_vec()
        })
    }

    pub fn receive(&mut self, mut keys_vec: Vec<std::os::raw::c_char>) -> ServerFHE {
        unsafe { server_keygen(keys_vec.as_mut_ptr()) }
    }
}

impl Drop for KeyShare {
    fn drop(&mut self) {
        unsafe {
            free_key_share(self.0);
        }
    }
}
