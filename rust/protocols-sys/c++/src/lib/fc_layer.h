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

Plaintext preprocess_vec(const Metadata& data, BatchEncoder& batch_encoder, const u64* input);

vector<Plaintext> preprocess_matrix(const u64* const* matrix, const Metadata& data, BatchEncoder& batch_encoder);

Plaintext fc_preprocess_noise(const Metadata& data, BatchEncoder& batch_encoder, const u64* secret_share);
    
Ciphertext fc_online(Ciphertext& ct, vector<Plaintext>& enc_mat, const Metadata& data, Evaluator& evaluator, GaloisKeys& gal_keys,
        RelinKeys& relin_keys, Ciphertext& zero);
   
u64* fc_postprocess(Ciphertext& result, const Metadata& data, BatchEncoder& batch_encoder, Decryptor& decryptor);

#endif
