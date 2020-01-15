/*
 *  Beaver's multiplication triple generation
 *
 *  Created on: August 10, 2019
 *      Author: ryanleh
 */
#ifndef triples_gen
#define triples_gen

#include "seal/seal.h"
#include "interface.h"
#include <boost/multiprecision/cpp_int.hpp>

using namespace seal;
using namespace std;

typedef uint64_t u64;
typedef boost::multiprecision::uint128_t u128;
typedef vector<u64> uv64;

vector<Ciphertext> encrypt_shares(const u64* share, int num_triples, Encryptor& encryptor, BatchEncoder& batch_encoder);

void server_compute_share(vector<Ciphertext> &a_ct, vector<Ciphertext> &b_ct,
        const u64* a_share, const u64* b_share, const u64* r_share, int num_triples,
        Evaluator& evaluator, BatchEncoder& batch_encoder, RelinKeys& relin_keys);
        
u64* client_share_postprocess(const u64* a_share, const u64* b_share,
        vector<Ciphertext> &client_share_ct, int num_triples, Evaluator& evaluator,
        BatchEncoder& batch_encoder, Decryptor& decryptor);

#endif
