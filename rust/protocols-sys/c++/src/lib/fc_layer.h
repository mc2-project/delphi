/*
 *  Gazelle's matrix multiplication ported to SEAL
 *
 *  Created on: June 10, 2019
 *      Author: ryanleh
 */
#ifndef fc_layer
#define fc_layer

#include "seal/seal.h"
#include "interface.h"
#include <math.h>

using namespace seal;
using namespace std;

typedef uint64_t u64;
typedef vector<u64> uv64;

Metadata fc_metadata(int slot_count, int vector_len, int matrix_h);

Ciphertext preprocess_vec(const u64* input, const Metadata& data, Encryptor& encryptor, BatchEncoder& batch_encoder);

vector<Plaintext> preprocess_matrix(const u64* const* matrix, const Metadata& data, BatchEncoder& batch_encoder);

Ciphertext fc_preprocess_noise(const u64* secret_share, const Metadata& data, Encryptor& encryptor, BatchEncoder& batch_encoder);
    
Ciphertext fc_online(Ciphertext& ct, vector<Plaintext>& enc_mat, const Metadata& data, Evaluator& evaluator, GaloisKeys& gal_keys,
        RelinKeys& relin_keys, Ciphertext& zero, Ciphertext& enc_noise);
   
u64* fc_postprocess(Ciphertext& result, const Metadata& data, BatchEncoder& batch_encoder, Decryptor& decryptor);

u64* HE_fc(u64* input, u64** matrix, int vector_len, int matrix_h);

uv64 fc_plain(u64* vec, u64** mat, int vector_len, int matrix_h);

#endif
