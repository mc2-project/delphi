#include <cassert>
#include <math.h>
#include "triples_gen.h"

/* Encrypts batch of triple shares */
vector<Ciphertext> encrypt_shares(const u64* share, int num_triples, Encryptor& encryptor,
        BatchEncoder& batch_encoder) {
    
    int slot_count = batch_encoder.slot_count();
    int num_ct = ceil((float)num_triples / slot_count);

    vector<Ciphertext> result(num_ct);
    
    int share_idx = 0;
    for (int ct_idx = 0; ct_idx < num_ct; ct_idx++) {
        Plaintext tmp;
        uv64 pod_matrix(slot_count, 0ULL);
        int limit = min(num_triples-share_idx, slot_count);
        for (int i = 0; i < limit; i++) {
            pod_matrix[i] = share[share_idx];
            share_idx++;
        }
        batch_encoder.encode(pod_matrix, tmp);
        encryptor.encrypt(tmp, result[ct_idx]);
    }

    return result;
}

/* Computes Enc(a1b2 + b1a2 + a2b2 + r), storing the result in a_ct */
void server_compute_share(vector<Ciphertext> &a_ct, vector<Ciphertext> &b_ct,
        const u64* a_share, const u64* b_share, const u64* r_share, int num_triples,
        Evaluator& evaluator, BatchEncoder& batch_encoder, RelinKeys& relin_keys) {
    
    int slot_count = batch_encoder.slot_count();
    int num_ct = a_ct.size();

    // Encode all shares to Plaintext for
    vector<Plaintext> a_enc(num_ct);
    vector<Plaintext> b_enc(num_ct);
    vector<Plaintext> c_enc(num_ct);

    int idx = 0;
    for (int ct_idx = 0; ct_idx < num_ct; ct_idx++) {
        uv64 pod1(slot_count, 0ULL);
        uv64 pod2(slot_count, 0ULL);
        uv64 pod3(slot_count, 0ULL);
        int limit = min(num_triples-idx, slot_count);
        for (int i = 0; i < limit; i++) {
            pod1[i] = a_share[idx];
            pod2[i] = b_share[idx];
            // Need to multiply into 128 bit int and cast down
            u128 product;
            product = boost::multiprecision::multiply(product, a_share[idx], b_share[idx]);
            pod3[i] = (u64)((product - r_share[idx]) % PLAINTEXT_MODULUS);
            idx++;
        }
        batch_encoder.encode(pod1, a_enc[ct_idx]);
        batch_encoder.encode(pod2, b_enc[ct_idx]);
        batch_encoder.encode(pod3, c_enc[ct_idx]);
    }
    for (int ct_idx = 0; ct_idx < num_ct; ct_idx++) {
        // Enc(a_1 b_2)
        evaluator.multiply_plain_inplace(a_ct[ct_idx], b_enc[ct_idx]);
        evaluator.relinearize_inplace(a_ct[ct_idx], relin_keys);
        // Enc(b_1 a_2)
        evaluator.multiply_plain_inplace(b_ct[ct_idx], a_enc[ct_idx]);
        evaluator.relinearize_inplace(b_ct[ct_idx], relin_keys);
        // Enc(a_1 b_2 + b_1 a_2)
        evaluator.add_inplace(a_ct[ct_idx], b_ct[ct_idx]);
        // Enc(a_1 b_2 + b_1 a_2 + a_2 b_2 - r)
        evaluator.add_plain_inplace(a_ct[ct_idx], c_enc[ct_idx]);
    }
}
        
u64* client_share_postprocess(const u64* a_share, const u64* b_share,
        vector<Ciphertext> &client_share_ct, int num_triples, Evaluator& evaluator,
        BatchEncoder& batch_encoder, Decryptor& decryptor) {

    u64* client_share = new u64[num_triples];
    int num_ct = client_share_ct.size();
    int slot_count = batch_encoder.slot_count();
    
    // Encode a_share * b_share
    vector<Plaintext> c_enc(num_ct);

    int idx = 0;
    for (int ct_idx = 0; ct_idx < num_ct; ct_idx++) {
        uv64 pod(slot_count, 0ULL);
        int limit = min(num_triples-idx, slot_count);
        for (int i = 0; i < limit; i++) {
            u128 product;
            boost::multiprecision::multiply(product, a_share[idx], b_share[idx]);
            pod[i] = (u64)(product % PLAINTEXT_MODULUS);
            idx++;
        }
        batch_encoder.encode(pod, c_enc[ct_idx]);
    }
    int result_idx = 0;
    for (int ct_idx = 0; ct_idx < num_ct; ct_idx++) {
        uv64 pod_matrix(slot_count, 0ULL);
        Plaintext tmp;
        evaluator.add_plain_inplace(client_share_ct[ct_idx], c_enc[ct_idx]);
        decryptor.decrypt(client_share_ct[ct_idx], tmp);
        batch_encoder.decode(tmp, pod_matrix);
        int limit = min(num_triples-result_idx, slot_count);
        for (int i = 0; i < limit; i++) {
            client_share[result_idx] = pod_matrix[i];
            result_idx++;
        }
    }
    return client_share;
}
