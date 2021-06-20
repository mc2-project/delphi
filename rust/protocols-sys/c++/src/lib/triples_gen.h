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

using namespace seal;
using namespace std;

typedef uint64_t u64;
typedef vector<u64> uv64;

/* Computes encrypted share of c ie. Enc(a1b1 + a1b2 + b1a2 + a2b2). Stores the result in c_ct.
 * Note that operations on a_ct and b_ct are done in-place for efficiency */
void triples_online(vector<Ciphertext> &a_ct, vector<Ciphertext> &b_ct, vector<Ciphertext> &c_ct,
        vector<Plaintext> &a_share, vector<Plaintext> &b_share, vector<Plaintext> &c_share,
        Evaluator& evaluator, RelinKeys& relin_keys);

/* Decrypts a ciphertext of triples shares */
u64* client_triples_postprocess(uint32_t num_triples, vector<Ciphertext> &ct, BatchEncoder &encoder,
    Decryptor& decryptor);

#endif
