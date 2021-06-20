/*
 *  FHE convolution taken from Gazelle (https://eprint.iacr.org/2018/073.pdf)
 *  with safer parameters and a few added optimizations for Delphi offline
 *  phase
 *
 *  Created on: June 10, 2019
 *      Author: ryanleh
 */
#ifndef conv2d
#define conv2d

#include "seal/seal.h"
#include "interface.h"

using namespace seal;
using namespace std;

typedef uint64_t u64;
typedef vector<u64> uv64;
typedef u64* Channel;
typedef Channel* Image;
typedef Image* Filters;

/* Rotate a plaintext vector in the correct half cyclic manner */
uv64 pt_rotate(int slot_count, int rotation, vector<u64> &vec);

/* Printing for Image */
void print(Image &image, int chans, int image_h, int image_w);

/* Print a Ciphertext */
void decrypt_and_print(Ciphertext, Metadata &data, Decryptor &decryptor, BatchEncoder &batch_encoder);

/* Helper function for performing modulo with possibly negative numbers */
inline int neg_mod(int val, int mod) {
    return ((val % mod) + mod) % mod;
}

/* Generate the Metadata struct */
Metadata conv_metadata(int slot_count, int image_h, int image_w, int filter_h, int filter_w,
        int inp_chans, int out_chans, int stride_h, int stride_w, bool pad_valid);

/* Generate noise to be applied to convolution result */
vector<Plaintext> HE_preprocess_noise(const u64* const* secret_share, const Metadata &data, BatchEncoder &batch_encoder);

/* Preprocesses the input image for output packing. Ciphertext is packed in RowMajor
 * order. In this mode simply pack all the input channels as tightly as possible
 * where each channel is padded to the nearest of two */
vector<uv64> preprocess_image(Metadata data, const u64* const* image);

/* Computes all the corresponding rotations of input according to the filter
 * dimensions
 * Note: have to pass a pointer instead of passing by reference to get the default parameter 
 * with NULL working properly */
template <class T> vector<T> filter_rotations(T &input, const Metadata &data, Evaluator *evaluator = NULL,
        GaloisKeys *gal_keys = NULL);

/* Encrypts all the input rotations */
vector<vector<Ciphertext>> HE_encrypt_rotations(vector<vector<uv64>> &rotations, const Metadata &data, Encryptor &encryptor,
        BatchEncoder &batch_encoder);

/* Encrypts the given input image */
vector<Ciphertext> HE_encrypt(vector<uv64> &pt, const Metadata &data, Encryptor &encryptor, BatchEncoder &batch_encoder);

/* Compute the necessary filters for an output packed input */
vector<vector<vector<Plaintext>>> HE_preprocess_filters(const u64* const* const* filters, const Metadata &data, BatchEncoder &batch_encoder);

/* Performs convolution for an output packed image. Returns the intermediate rotation sets */
vector<vector<Ciphertext>> HE_conv(vector<vector<vector<Plaintext>>> &masks, vector<vector<Ciphertext>> &rotations,
        const Metadata &data, Evaluator &evaluator, RelinKeys &relin_keys, Ciphertext &zero);

/* Rotates and adds an output packed convolution result to produce a final, tight output */
vector<Ciphertext> HE_output_rotations(vector<vector<Ciphertext>> convs, const Metadata &data, Evaluator &evaluator,
        GaloisKeys &gal_keys, Ciphertext &zero);

/* Decrypts and reshapes convolution result */
u64** HE_decrypt(vector<Ciphertext> &enc_result, const Metadata &data, Decryptor &decryptor, BatchEncoder &batch_encoder);
#endif
