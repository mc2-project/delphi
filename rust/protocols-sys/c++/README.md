A re-implementation of [Gazelle's](https://github.com/chiraag/gazelle_mpc) homomorphic protocols for convolution and matrix multiplication with minor improvements in SEAL.

## Building
The [protocols-sys](../rust/protocols-sys) folder should handle the entire build process.

However, if manual building is necessary, the base library requires make, cmake, gcc, and boost. Optionally, unittests can be built by installing [Eigen3.3](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download) and passing the `-DUNITTESTS=1` flag to `cmake`.

The latest compatible version (3.3.2) of [SEAL](https://github.com/microsoft/SEAL) is included in the repo for convenience. If you wish to link to a different version of SEAL, pass the  `-DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=0` flag to `cmake`. By default, SEAL throws exceptions on transparent ciphertexts, but Delphi makes intermediate use of them in a secure manner.

Note that manual builds may cause issues with the automatic Rust building. Most of these problems can be avoided by removing any CMake cache files in between manual and automated builds.
