#include <iomanip>
#include <cassert>
#include <math.h>
#include "fc_layer.h"

/* Helper function for rounding to the next power of 2
 * Credit: https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2 */
inline int next_pow2(int val) {
    return pow(2, ceil(log(val)/log(2)));
}

/* Formatted printed for batched plaintext */
inline void print_batch(size_t slot_count, size_t print_size, uv64 &plain) {
    size_t row_size = slot_count / 2;
    cout << endl;

    cout << "    [";
    for (size_t i = 0; i < print_size; i++)
    {
        cout << setw(3) << plain[i] << ",";
    }
    cout << setw(3) << " ...,";
    for (size_t i = row_size - print_size; i < row_size; i++)
    {
        cout << setw(3) << plain[i] << ((i != row_size - 1) ? "," : " ]\n");
    }
    cout << "    [";
    for (size_t i = row_size; i < row_size + print_size; i++)
    {
        cout << setw(3) << plain[i] << ",";
    }
    cout << setw(3) << " ...,";
    for (size_t i = 2 * row_size - print_size; i < 2 * row_size; i++)
    {
        cout << setw(3) << plain[i] << ((i != 2 * row_size - 1) ? "," : " ]\n");
    }
    cout << endl;
}

/* Helper function to print Ciphertexts */
inline void decrypt_and_print(Ciphertext ct, Metadata &data, Decryptor &decryptor,
        BatchEncoder &batch_encoder) {
    Plaintext tmp;
    vector<u64> pod_matrix(data.slot_count, 0ULL);
    decryptor.decrypt(ct, tmp);
    batch_encoder.decode(tmp, pod_matrix);
    cout << endl;
    for (int chan = 0; chan < 1; chan++) {
        for (int i = 0; i < 1024; i++)
            cout << pod_matrix[chan*data.image_size + i] << ", ";
        cout << " || ";
    }
    cout << endl;
}


/* We reuse the convolution struct where image=vector and filter=matrix */
Metadata fc_metadata(int slot_count, int vector_len, int matrix_h) {
    Metadata data;
    data.slot_count = slot_count;
    data.image_h = 1;
    data.image_w = vector_len;
    data.image_size = vector_len;
    data.filter_h = matrix_h;
    data.filter_w = vector_len;
    data.filter_size =  data.filter_h * data.filter_w;
    // How many columns of matrix we can fit in a single ciphertext
    data.pack_num = slot_count / next_pow2(data.filter_w);
    // How many total ciphertexts we'll need
    data.inp_ct = ceil((float)next_pow2(data.filter_h) / data.pack_num);
    
    return data;
}

Ciphertext preprocess_vec(const u64* input, const Metadata& data,
        Encryptor& encryptor, BatchEncoder& batch_encoder) {
    // Only works with a ciphertext that fits in a single ciphertext
    assert(data.slot_count >= data.image_size);
    
    // Create copies of the input vector to fill the ciphertext appropiately.
    // Pack using powers of two for easy rotations later
    uv64 pod_matrix(data.slot_count, 0ULL);
    u64 size_pow2 = next_pow2(data.image_size);
    for(int col = 0; col < data.image_size; col++){
        for(int idx = 0; idx < data.pack_num; idx++){
            pod_matrix[col + size_pow2 * idx] = input[col];
        }
    }
    
    // Encrypt plaintext
    Ciphertext ciphertext;
    Plaintext tmp;
    batch_encoder.encode(pod_matrix, tmp);
    encryptor.encrypt(tmp, ciphertext);
    return ciphertext;
}

vector<Plaintext> preprocess_matrix(const u64* const* matrix,
        const Metadata& data, BatchEncoder& batch_encoder) {
    // Pack the filter in alternating order of needed ciphertexts. This way we
    // rotate the input once per ciphertext
    vector<uv64> mat_pack(data.inp_ct, uv64(data.slot_count, 0ULL));
    for(int row = 0; row < data.filter_h; row++){
        int ct_idx = row / data.inp_ct;
        for(int col = 0; col < data.filter_w; col++){
            mat_pack[row%data.inp_ct][col+next_pow2(data.filter_w)*ct_idx] = matrix[row][col];
        }
    }

    // Take the packed ciphertexts above and repack them in a diagonal ordering. 
    int mod_mask = (data.inp_ct-1);
    int wrap_thresh = min(data.slot_count >> 1, next_pow2(data.filter_w));
    int wrap_mask = wrap_thresh - 1;
    vector<uv64> mat_diag(data.inp_ct, uv64(data.slot_count, 0ULL));
    for(int ct = 0; ct < data.inp_ct; ct++){
        for(int col = 0; col < data.slot_count; col++){
            int ct_diag_l = (col-ct) & wrap_mask & mod_mask;
            int ct_diag_h = (col^ct) & (data.slot_count/2) & mod_mask;
            int ct_diag = (ct_diag_h + ct_diag_l);

            int col_diag_l = (col - ct_diag_l) & wrap_mask;
            int col_diag_h = wrap_thresh*(col/wrap_thresh) ^ ct_diag_h;
            int col_diag = col_diag_h + col_diag_l;

            mat_diag[ct_diag][col_diag] = mat_pack[ct][col];
        }
    }
    
    // Encode matrix vectors
    vector<Plaintext> enc_mat(data.inp_ct);
    for (int ct = 0; ct < data.inp_ct; ct++) {
        batch_encoder.encode(mat_diag[ct], enc_mat[ct]);
    }
    return enc_mat;
}


/* Generates a masking vector of random noise that will be applied to parts of the ciphertext
 * that contain leakage */
Ciphertext fc_preprocess_noise(const uint64_t* secret_share, const Metadata &data,
        Encryptor& encryptor, BatchEncoder& batch_encoder) {
    // Create uniform distribution
    random_device rd;
    mt19937 engine(rd());
    // TODO: Tune this
    uniform_int_distribution<u64> dist(0, 1<<20);
    auto gen = [&dist, &engine](){
        return dist(engine);
    };
    
    // Sample randomness into vector
    uv64 noise(data.slot_count, 0ULL);
    generate(begin(noise), end(noise), gen);

    // Puncture the vector with secret shares where an actual fc result value lives
    for (int row = 0; row < data.filter_h; row++) {
        int curr_set = row / data.inp_ct;
        noise[(row%data.inp_ct)+next_pow2(data.image_size)*curr_set] = secret_share[row];
    }

    // Encrypt the noise vector
    Ciphertext enc_noise;
    Plaintext tmp;
    batch_encoder.encode(noise, tmp);
    encryptor.encrypt(tmp, enc_noise);
    return enc_noise; 
}


Ciphertext fc_online(Ciphertext& ct, vector<Plaintext>& enc_mat,
        const Metadata& data, Evaluator& evaluator, GaloisKeys& gal_keys,
        RelinKeys& relin_keys, Ciphertext& zero, Ciphertext& enc_noise) {
    Ciphertext result = zero;
    // For each matrix ciphertext, rotate the input vector once and multiply +
    // add
    for(int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
        Ciphertext tmp;
        evaluator.rotate_rows(ct, ct_idx, gal_keys, tmp);
        evaluator.multiply_plain_inplace(tmp, enc_mat[ct_idx]);
        evaluator.relinearize_inplace(tmp, relin_keys);
        evaluator.add_inplace(result, tmp);
    }

    // Rotate all partial sums together
    for (int rot = data.inp_ct; rot < next_pow2(data.image_size); rot *= 2){
        Ciphertext tmp;
        if (rot == data.slot_count/2) {
            evaluator.rotate_columns(result, gal_keys, tmp);
        } else {
            evaluator.rotate_rows(result, rot, gal_keys, tmp);
        }
        evaluator.add_inplace(result, tmp);
    }
    // Add noise to cover leakage
    evaluator.add_inplace(result, enc_noise);
    return result;
}

u64* fc_postprocess(Ciphertext& ct, const Metadata& data, BatchEncoder& batch_encoder, Decryptor& decryptor) {
    // Decrypt + decode ciphertext
    uv64 plain(data.slot_count, 0ULL);
    Plaintext tmp;
    decryptor.decrypt(ct, tmp);
    batch_encoder.decode(tmp, plain);

    // Grab the appropiate indices from the result
    u64* result = new u64[data.filter_h];
    for (int row = 0; row < data.filter_h; row++) {
        int curr_set = row / data.inp_ct;
        result[row] = plain[(row % data.inp_ct) + next_pow2(data.image_size)*curr_set];
    }
    return result;
}

/* Example use of fc functions */
u64* HE_fc(u64* input, u64** matrix, int vector_len, int matrix_h) {
    //---------------Param and Key Generation---------------
    EncryptionParameters parms(scheme_type::BFV);
    parms.set_poly_modulus_degree(POLY_MOD_DEGREE);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(POLY_MOD_DEGREE));
    parms.set_plain_modulus(PLAINTEXT_MODULUS);
    auto context = SEALContext::Create(parms);
    KeyGenerator keygen(context);
    auto public_key = keygen.public_key();
    auto secret_key = keygen.secret_key();
    // Parameters are large enough that we should be fine with these at max
    // decomposition
    auto relin_keys = keygen.relin_keys();
    auto gal_keys = keygen.galois_keys();

    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key); 
    BatchEncoder batch_encoder(context);

    int slot_count = batch_encoder.slot_count();
    // Generate the zero ciphertext
    vector<u64> pod_matrix(slot_count, 0ULL);
    Plaintext tmp;
    Ciphertext zero;
    batch_encoder.encode(pod_matrix, tmp);
    encryptor.encrypt(tmp, zero);
    //-------------------------------------------
    auto data = fc_metadata(slot_count, vector_len, matrix_h);

    u64* secret_share = new u64[vector_len];
    for (int i = 0; i < vector_len; i++)
        secret_share[i] = 0;

    // Generate noise
    Ciphertext enc_noise = fc_preprocess_noise(secret_share, data, encryptor, batch_encoder);

    auto ct = preprocess_vec(input, data, encryptor, batch_encoder);
    auto enc_mat = preprocess_matrix(matrix, data, batch_encoder);

    auto result = fc_online(ct, enc_mat, data, evaluator, gal_keys, relin_keys, zero, enc_noise);

    auto final_result = fc_postprocess(result, data, batch_encoder, decryptor);

    return final_result;
}

uv64 fc_plain(u64* input, u64** matrix, int vector_len, int matrix_h) {
    uv64 result(matrix_h, 0ULL);
    for (int row = 0; row < matrix_h; row++){
        for (int idx = 0; idx < vector_len; idx++){
            u64 partial = input[idx]*matrix[row][idx];
            result[row] = result[row] + partial;
        }
    }
    return result;
}

