#include <cassert>
#include <math.h>
#include "triples_gen.h"

void triples_online(vector<Ciphertext> &a_ct, vector<Ciphertext> &b_ct, vector<Ciphertext> &c_ct,
        vector<Plaintext> &a_rand, vector<Plaintext> &b_rand, vector<Plaintext> &c_rand,
        Evaluator& evaluator, RelinKeys& relin_keys) {
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int ct_idx = 0; ct_idx < a_ct.size(); ct_idx++) {
        // Enc(a_1 b_1) -> c_ct
        evaluator.multiply(a_ct[ct_idx], b_ct[ct_idx], c_ct[ct_idx]);
        evaluator.relinearize_inplace(c_ct[ct_idx], relin_keys);
        // Enc(a_1 b_2)
        evaluator.multiply_plain_inplace(a_ct[ct_idx], b_rand[ct_idx]);
        // Enc(b_1 a_2)
        evaluator.multiply_plain_inplace(b_ct[ct_idx], a_rand[ct_idx]);
        // Enc(a_1 b_1 + a_1 b_2)
        evaluator.add_inplace(c_ct[ct_idx], a_ct[ct_idx]);
        // Enc(a_1 b_1 + a_1 b_2 + b_1 a_2)
        evaluator.add_inplace(c_ct[ct_idx], b_ct[ct_idx]);
        // Enc(a_1 b_1 + a_1 b_2 + b_1 a_2 + a_2 b_2 + r) = Enc(c + r)
        evaluator.add_plain_inplace(c_ct[ct_idx], c_rand[ct_idx]);
    }
}
        
u64* client_triples_postprocess(uint32_t num_triples, vector<Ciphertext> &ct, BatchEncoder& encoder,
        Decryptor& decryptor) {
    uint32_t slot_count = encoder.slot_count();
    // Allocate space for resulting plaintext
    u64* share = new u64[num_triples];
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int ct_idx = 0; ct_idx < ct.size(); ct_idx++) {
        uv64 pod_matrix(slot_count, 0ULL);
        Plaintext tmp;
        decryptor.decrypt(ct[ct_idx], tmp);
        encoder.decode(tmp, pod_matrix);
        int limit = min(num_triples-ct_idx*slot_count, slot_count);
        for (int i = 0; i < limit; i++) {
            share[ct_idx*slot_count + i] = pod_matrix[i];
        }
    }
    return share;
}
