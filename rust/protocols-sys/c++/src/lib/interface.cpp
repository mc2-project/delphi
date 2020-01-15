/*
 *  DelphiOffline's C interface 
 *
 *  Created on: June 10, 2019
 *      Author: ryanleh
 */
#include "interface.h"
#include "conv2d.h"
#include "fc_layer.h"
#include "triples_gen.h"

/* Generates new keys and helpers for Client. Returns the helpers and allocates
 * a bytestream for sharing the keys with the server */
ClientFHE client_keygen(char** key_share, uint64_t* key_share_len) {
    //---------------Param and Key Generation---------------
    EncryptionParameters parms(scheme_type::BFV);
    parms.set_poly_modulus_degree(POLY_MOD_DEGREE);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(POLY_MOD_DEGREE));
    parms.set_plain_modulus(PLAINTEXT_MODULUS);
    auto context = SEALContext::Create(parms);
    KeyGenerator keygen(context);
    auto pub_key = keygen.public_key();
    auto sec_key = keygen.secret_key();
    // Parameters are large enough that we should be fine with these at max
    // decomposition
    auto gal_keys = keygen.galois_keys();
    auto relin_keys = keygen.relin_keys();

    BatchEncoder *encoder = new BatchEncoder(context);
    Encryptor *encryptor = new Encryptor(context, pub_key);
    Evaluator *evaluator = new Evaluator(context);
    Decryptor *decryptor = new Decryptor(context, sec_key); 
   
    // Create a byte array consisting of the three serialized keys and their
    // sizes
    ostringstream os;
    // Public Key
    pub_key.save(os);
    uint64_t pk_size = os.tellp();
    // Galois Keys
    gal_keys.save(os);
    uint64_t gk_size = (uint64_t)os.tellp()-pk_size;
    // Relin keys
    relin_keys.save(os);
    uint64_t rk_size = (uint64_t)os.tellp()-pk_size-gk_size;

    string keys_ser = os.str();
    *key_share_len = sizeof(uint64_t)*3+pk_size+gk_size+rk_size;
    *key_share = new char[*key_share_len];
    // std::copy does weird stuff with typing for POD so have to use memcpy here
    // Copy the size of each key followed by the key itself
    memcpy(*key_share, &pk_size, sizeof(uint64_t));
    memcpy(*key_share+sizeof(uint64_t), &gk_size, sizeof(uint64_t));
    memcpy(*key_share+sizeof(uint64_t)*2, &rk_size, sizeof(uint64_t));
    std::move(keys_ser.begin(), keys_ser.end(), *key_share+sizeof(uint64_t)*3);

    return ClientFHE { encoder, encryptor, evaluator, decryptor };
}

/* Generates keys and helpers for Server given a key_share */
ServerFHE server_keygen(char* key_share) {
    //---------------Param Generation---------------
    EncryptionParameters parms(scheme_type::BFV);

    parms.set_poly_modulus_degree(POLY_MOD_DEGREE);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(POLY_MOD_DEGREE));
    parms.set_plain_modulus(PLAINTEXT_MODULUS);
    auto context = SEALContext::Create(parms);

    // Extract the size of each key and then put the key into an inputstream so
    // that it can be loaded into the correct object
    istringstream is;

    uint64_t pk_size;
    uint64_t gk_size;
    uint64_t rk_size;
    memcpy(&pk_size, key_share, sizeof(uint64_t));
    memcpy(&gk_size, key_share+sizeof(uint64_t), sizeof(uint64_t));
    memcpy(&rk_size, key_share+sizeof(uint64_t)*2, sizeof(uint64_t));

    // Public Key
    PublicKey pub_key;
    is.rdbuf()->pubsetbuf(key_share+sizeof(uint64_t)*3, pk_size);
    pub_key.load(context, is);
    // Galois Keys 
    GaloisKeys* gal_keys = new GaloisKeys();
    is.rdbuf()->pubsetbuf(key_share+sizeof(uint64_t)*3+pk_size, gk_size);
    (*gal_keys).load(context, is);
    // Relin Keys
    RelinKeys* relin_keys = new RelinKeys();
    is.rdbuf()->pubsetbuf(key_share+sizeof(uint64_t)*3+pk_size+gk_size, rk_size);
    (*relin_keys).load(context, is);
    
    BatchEncoder *encoder = new BatchEncoder(context);
    Encryptor *encryptor = new Encryptor(context, pub_key);
    Evaluator *evaluator = new Evaluator(context);
    
    // Generate the zero ciphertext
    vector<uint64_t> pod_matrix(encoder->slot_count(), 0ULL);
    Plaintext tmp;
    Ciphertext* zero = new Ciphertext();
    encoder->encode(pod_matrix, tmp);
    encryptor->encrypt(tmp, *zero);

    return ServerFHE { encoder, encryptor, evaluator, gal_keys, relin_keys, (char*) zero };
}

/* Populates the Metadata struct */
Metadata conv_metadata(void* batch_encoder, int32_t image_h, int32_t image_w, int32_t filter_h, int32_t filter_w,
        int32_t inp_chans, int32_t out_chans, int32_t stride_h, int32_t stride_w, bool pad_valid, Mode mode) {
    int slot_count = (reinterpret_cast<BatchEncoder*>(batch_encoder))->slot_count();
    return conv_metadata(slot_count, image_h, image_w, filter_h, filter_w, inp_chans, out_chans,
        stride_h, stride_w, pad_valid, mode);
}

/* Populates the Metadata struct for FC layer */
Metadata fc_metadata(void* batch_encoder, int32_t vector_len, int32_t matrix_h) {
    int slot_count = (reinterpret_cast<BatchEncoder*>(batch_encoder))->slot_count();
    return fc_metadata(slot_count, vector_len, matrix_h);
}

/* Allocates correct serialized objects for each provided point32_ter */
char* client_conv_preprocess(const uint64_t* const* image, const ClientFHE* cfhe, const Metadata* data, Mode mode, uint64_t* enc_size) {
    // Recast the needed fhe helpers
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(cfhe->encryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    
    // Populate the image ciphertext based on the mode being used
    vector<vector<Ciphertext>> ct_vec;
    if (mode == Mode::InputConv) {
        auto pt = preprocess_image_IP(image, *data);
        auto rotated_pt = filter_rotations(pt, *data);
        ct_vec = HE_encrypt_rotations(rotated_pt, *data, *encryptor, *encoder);
    } else if (mode == Mode::Input) {
        auto pt = preprocess_image_IP(image, *data);
        ct_vec.resize(1);
        ct_vec[0] = HE_encrypt(pt, *data, *encryptor, *encoder);
    } else if (mode == Mode::Output) {
        auto pt = preprocess_image_OP(image, *data);
        auto rotated_pt = filter_rotations(pt, *data);
        ct_vec = HE_encrypt_rotations(rotated_pt, *data, *encryptor, *encoder);
    }

    // Convert vector to char array and flatten to a single byte array
    ostringstream os;
    uint64_t ct_size = 0;
    for (int ct_idx = 0; ct_idx < ct_vec.size(); ct_idx++) {
        for(int rot = 0; rot < ct_vec[0].size(); rot++) {
            ct_vec[ct_idx][rot].save(os);
            if (!ct_idx && !rot)
                ct_size = os.tellp();
        }
    }
    string ct_ser = os.str();
    char* ciphertext = new char[sizeof(uint64_t) + ct_ser.size()];
    *enc_size = sizeof(uint64_t) + ct_ser.size();
    memcpy(ciphertext, &ct_size, sizeof(uint64_t));
    std::move(ct_ser.begin(), ct_ser.end(), ciphertext+sizeof(uint64_t));
    return ciphertext;
}

/* Generates necessary filters and stores in tensor */
char**** server_conv_preprocess(const uint64_t* const* const* filters, const ServerFHE* sfhe,
        const Metadata* data, Mode mode) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);

    // Create the filters
    vector<vector<vector<Plaintext>>> masks_vec;
    if (mode == Mode::InputConv) {
        masks_vec.resize(1);
        masks_vec[0] = HE_preprocess_filters_IP(filters, *data, *encoder);
    } else if (mode == Mode::Input) {
        masks_vec.resize(1);
        masks_vec[0] = HE_preprocess_filters_IP(filters, *data, *encoder);
    } else if (mode == Mode::Output) {
        masks_vec = HE_preprocess_filters_OP(filters, *data, *encoder);
    }
    
    char**** masks = new char***[masks_vec.size()];
    for (int i = 0; i < masks_vec.size(); i++) {
        masks[i] = new char**[masks_vec[0].size()];
        for (int j = 0; j < masks_vec[0].size(); j++) {
            masks[i][j] = new char*[masks_vec[0][0].size()];
            for (int z = 0; z < masks_vec[0][0].size(); z++)
                masks[i][j][z] = (char*) new Plaintext(masks_vec[i][j][z]);
        }
    }

    return masks;
}
 
/* Preprocess and serializes client input vector */
char* client_fc_preprocess(const uint64_t* vector, const ClientFHE* cfhe, const Metadata* data, uint64_t* enc_size) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(cfhe->encryptor);

    auto ciphertext = preprocess_vec(vector, *data, *encryptor, *encoder);

    // Convert vector to char array and flatten to a single byte array
    ostringstream os;
    ciphertext.save(os);
    uint64_t ct_size = os.tellp();
    
    string ct_ser = os.str();
    char* ciphertext_c = new char[sizeof(uint64_t) + ct_size];
    *enc_size = sizeof(uint64_t) + ct_size;
    memcpy(ciphertext_c, &ct_size, sizeof(uint64_t));
    std::move(ct_ser.begin(), ct_ser.end(), ciphertext_c+sizeof(uint64_t));
    return ciphertext_c;
}
    
/* Preprocesses server matrix and enc_noise */
char** server_fc_preprocess(const uint64_t* const* matrix, const ServerFHE* sfhe, const Metadata* data) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);

    // Create the filters
    vector<Plaintext> enc_matrix = preprocess_matrix(matrix, *data, *encoder);
   
    // Convert to opaque C types
    char** enc_matrix_c = new char*[data->inp_ct];
    for (int i = 0; i < data->inp_ct; i++) {
        enc_matrix_c[i] = (char*) new Plaintext(enc_matrix[i]);
    }
    return enc_matrix_c;
}

/* Encrypts a triple share */
char* triples_preprocess(const uint64_t* share, void* encoder_c, void* encryptor_c,
        int num_triples, uint64_t* enc_size) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(encoder_c);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(encryptor_c);

    auto share_cts = encrypt_shares(share, num_triples, *encryptor, *encoder);
    int num_ct = share_cts.size();

    // Convert vector to char array and flatten to a single byte array
    ostringstream os;
    uint64_t ct_size = 0;
    for (int ct_idx = 0; ct_idx < share_cts.size(); ct_idx++) {
        share_cts[ct_idx].save(os);
        if (!ct_idx)
            ct_size = os.tellp();
    }
    string ct_ser = os.str();
    char* result = new char[sizeof(uint64_t) + ct_ser.size()];
    *enc_size = sizeof(uint64_t) + ct_ser.size();
    memcpy(result, &ct_size, sizeof(uint64_t));
    std::move(ct_ser.begin(), ct_ser.end(), result+sizeof(uint64_t));
    return result;
}

/* Computes the masking noise ciphertext for conv output */        
char** conv_preprocess_noise(const ServerFHE* sfhe, const Metadata* data, const uint64_t* const* secret_share) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(sfhe->encryptor);

    // Generate noise ciphertext
    vector<Ciphertext> noise = HE_preprocess_noise(secret_share, *data, *encryptor, *encoder);

    // Convert everything back to opaque C types
    auto enc_noise = new char*[data->out_ct];
    for (int ct_idx = 0; ct_idx < data->out_ct; ct_idx++) {
        enc_noise[ct_idx] = (char*) new Ciphertext(noise[ct_idx]);
    }
    return enc_noise;
}

/* Computes the masking noise ciphertext for conv output */
char* fc_preprocess_noise(const ServerFHE* sfhe, const Metadata* data, const uint64_t* secret_share) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(sfhe->encryptor);

    // Generate noise ciphertext
    Ciphertext noise = fc_preprocess_noise(secret_share, *data, *encryptor, *encoder);

    // Convert everything back to opaque C types
    auto enc_noise = (char*) new Ciphertext(noise);
    return enc_noise;
}

/* Performs the convolution on the given input */
char* server_conv_online(char* ciphertext, char**** masks, const ServerFHE *sfhe, const Metadata *data,
        Mode mode, char** enc_noise, uint64_t* enc_result_size) {
    vector<vector<Ciphertext>> ct_vec;
    vector<vector<vector<Plaintext>>> masks_vec;
    vector<Ciphertext> noise_ct(data->out_ct);
    // Resize the vectors to be the correct capacity depending on the mode
    if (mode == Mode::InputConv) {
        ct_vec.resize(data->inp_ct, vector<Ciphertext>(data->filter_size));
        masks_vec.resize(1, 
                vector<vector<Plaintext>>(data->inp_ct, 
                    vector<Plaintext>(data->filter_size)));
    } else if (mode == Mode::Input) {
        ct_vec.resize(1, vector<Ciphertext>(data->inp_ct));
        masks_vec.resize(1, 
                vector<vector<Plaintext>>(data->inp_ct, 
                    vector<Plaintext>(data->filter_size)));
    } else if (mode == Mode::Output) {
        ct_vec.resize(data->inp_ct, vector<Ciphertext>(data->filter_size));
        masks_vec.resize(data->convs, 
                vector<vector<Plaintext>>(data->inp_ct, 
                    vector<Plaintext>(data->filter_size)));
    }
    uint64_t ct_size;
    memcpy(&ct_size, ciphertext, sizeof(uint64_t));
    istringstream is;
    // Remap raw pointers to vectors
    for (int i = 0; i < ct_vec.size(); i++) {
        for (int j = 0; j < ct_vec[0].size(); j++) {
            u64 idx = sizeof(uint64_t) + (i*ct_vec[0].size() + j) * ct_size;
            is.rdbuf()->pubsetbuf(ciphertext + idx, ct_size);
            ct_vec[i][j].unsafe_load(is);
        } 
    }
    
    for (int i = 0; i < masks_vec.size(); i++) {
        for (int j = 0; j < masks_vec[0].size(); j++) {
            for (int z = 0; z < masks_vec[0][0].size(); z++)
                masks_vec[i][j][z] = *(reinterpret_cast<Plaintext*>(masks[i][j][z]));
        } 
    }

    for (int ct_idx = 0; ct_idx < data->out_ct; ct_idx++) {
        noise_ct[ct_idx] = *(reinterpret_cast<Ciphertext*>(enc_noise[ct_idx]));
    }
    // Recast needed fhe helpers and ciphertexts
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(sfhe->encryptor);
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(sfhe->evaluator);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    GaloisKeys *gal_keys = reinterpret_cast<GaloisKeys*>(sfhe->gal_keys);
    RelinKeys *relin_keys = reinterpret_cast<RelinKeys*>(sfhe->relin_keys);
    Ciphertext *zero = reinterpret_cast<Ciphertext*>(sfhe->zero);

    // Evaluation
    vector<Ciphertext> result;
    if (mode == Mode::InputConv) {
        result = HE_conv_IP(masks_vec[0], ct_vec, *data, *evaluator, *relin_keys, *zero, noise_ct);
    } else if (mode == Mode::Input) {
        // Generate rotations before convolution
        ct_vec = filter_rotations(ct_vec[0], *data, evaluator, gal_keys);
        result = HE_conv_IP(masks_vec[0], ct_vec, *data, *evaluator, *relin_keys, *zero, noise_ct);
    } else if (mode == Mode::Output) {
        auto rotation_sets = HE_conv_OP(masks_vec, ct_vec, *data, *evaluator, *relin_keys, *zero); 
        result = HE_output_rotations(rotation_sets, *data, *evaluator, *gal_keys, *zero, noise_ct);
    }

    // Serialize the resuling ciphertexts and pack them into a byte array to
    // be sent to client
    ostringstream os;
    uint64_t result_size = sizeof(uint64_t) + ct_size*data->out_ct;
    *enc_result_size = result_size;
    char* result_c = new char[result_size];
    memcpy(result_c, &ct_size, sizeof(uint64_t));
    // Add the remaining ciphertext to the output stream
    for(int ct = 0; ct < data->out_ct; ct++) {
        result[ct].save(os);
    }
    // Move the output stream to the byte array
    string ct_ser = os.str();
    std::move(ct_ser.begin(), ct_ser.end(), result_c+sizeof(uint64_t));
    return result_c;
}

/* Performs matrix multiplication on the given inputs */
char* server_fc_online(char* ciphertext, char** enc_matrix, const ServerFHE* sfhe, const Metadata* data,
        char* enc_noise, uint64_t* enc_result_size) {
    uint64_t ct_size;
    memcpy(&ct_size, ciphertext, sizeof(uint64_t));
    istringstream is;
    // Remap raw pointers to vectors
    Ciphertext input;
    vector<Plaintext> matrix_vec(data->inp_ct);
    Ciphertext noise_ct;
    
    is.rdbuf()->pubsetbuf(ciphertext + sizeof(uint64_t), ct_size);
    input.unsafe_load(is);
    for (int i = 0; i < data->inp_ct; i++)
        matrix_vec[i] = *(reinterpret_cast<Plaintext*>(enc_matrix[i]));
    noise_ct = *(reinterpret_cast<Ciphertext*>(enc_noise));

    // Recast needed fhe helpers and ciphertexts
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(sfhe->encryptor);
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(sfhe->evaluator);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    GaloisKeys *gal_keys = reinterpret_cast<GaloisKeys*>(sfhe->gal_keys);
    RelinKeys *relin_keys = reinterpret_cast<RelinKeys*>(sfhe->relin_keys);
    Ciphertext *zero = reinterpret_cast<Ciphertext*>(sfhe->zero);

    // Evaluation
    Ciphertext result = fc_online(input, matrix_vec, *data, *evaluator, *gal_keys, *relin_keys, *zero, noise_ct);

    // Serialize the resuling ciphertext and pack it into a byte array to
    // be sent to client
    ostringstream os;
    *enc_result_size = sizeof(uint64_t) + ct_size;
    char* result_c = new char[*enc_result_size];
    memcpy(result_c, enc_result_size, sizeof(uint64_t));
    // Move the output stream to the byte array
    result.save(os);
    string ct_ser = os.str();
    std::move(ct_ser.begin(), ct_ser.end(), result_c+sizeof(uint64_t));
    return result_c;
}

/* Computes the encrypted client's share of multiplication triple */
char* server_triples_online(char* client_a, char* client_b, const uint64_t* a_share,
        const uint64_t* b_share, const uint64_t* r_share, const ServerFHE* sfhe,
        int num_triples, uint64_t* enc_size) {
    // Recast needed fhe helpers and ciphertexts
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(sfhe->evaluator);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    RelinKeys *relin_keys = reinterpret_cast<RelinKeys*>(sfhe->relin_keys);

    int slot_count = encoder->slot_count();
    int num_ct = ceil((float)num_triples/slot_count);

    // Remap raw pointers to vectors
    vector<Ciphertext> a_ct(num_ct);
    vector<Ciphertext> b_ct(num_ct);
    
    uint64_t ct_size;
    memcpy(&ct_size, client_a, sizeof(uint64_t));

    istringstream is;
    for (int ct_idx = 0; ct_idx < num_ct; ct_idx++) {
        u64 idx = sizeof(uint64_t) + ct_idx*ct_size;
        is.rdbuf()->pubsetbuf(client_a + idx, ct_size);
        a_ct[ct_idx].unsafe_load(is);
        is.rdbuf()->pubsetbuf(client_b + idx, ct_size);
        b_ct[ct_idx].unsafe_load(is);
    }

    // Evaluation
    server_compute_share(a_ct, b_ct, a_share, b_share, r_share, num_triples, *evaluator, *encoder, *relin_keys);

    // Serialize the resuling ciphertext and pack it into a byte array to
    // be sent to client
    ostringstream os;
    for (int ct_idx = 0; ct_idx < num_ct; ct_idx++) {
        a_ct[ct_idx].save(os);
    }
    string ct_ser = os.str();
    char* client_share_c = new char[sizeof(uint64_t) + ct_ser.size()];
    *enc_size = sizeof(uint64_t) + ct_ser.size();
    memcpy(client_share_c, &ct_size, sizeof(uint64_t));
    std::move(ct_ser.begin(), ct_ser.end(), client_share_c+sizeof(uint64_t));
    return client_share_c;
}

/* Decrypts and reshapes convolution result */
uint64_t** client_conv_decrypt(char* c_enc_result, const ClientFHE* cfhe, const Metadata* data) {
    // Recast needed fhe helpers
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(cfhe->decryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);

    // Map byte stream to vector of Ciphertexts
    vector<Ciphertext> enc_result(data->out_ct);
    istringstream is;
    uint64_t ct_size;
    memcpy(&ct_size, c_enc_result, sizeof(uint64_t));
    for(int ct = 0; ct < data->out_ct; ct++) {
        is.str(std::string());
        is.rdbuf()->pubsetbuf(c_enc_result+sizeof(uint64_t)+ct_size*ct, ct_size);
        enc_result[ct].unsafe_load(is);
    }

    auto result = HE_decrypt(enc_result, *data, *decryptor, *encoder);
    return result;
}

/* Decrypts and reshapes fully-connected result */
uint64_t* client_fc_decrypt(char* enc_result, const ClientFHE *cfhe, const Metadata *data) {
    // Recast needed fhe helpers
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(cfhe->decryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);

    // Map byte stream to Ciphertext
    Ciphertext result_ct;
    istringstream is;
    uint64_t ct_size;
    memcpy(&ct_size, enc_result, sizeof(uint64_t));
    is.rdbuf()->pubsetbuf(enc_result+sizeof(uint64_t), ct_size);
    result_ct.unsafe_load(is);

    auto result = fc_postprocess(result_ct, *data, *encoder, *decryptor);
    return result;
}

/* Decrypts the clients multiplication triple share */
uint64_t* client_triples_decrypt(const u64* a_share, const u64* b_share, char* client_share,
        const ClientFHE* cfhe, int num_triples) {
    // Recast needed fhe helpers
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(cfhe->decryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(cfhe->evaluator);

    int slot_count = encoder->slot_count();
    int num_ct = ceil((float)num_triples/slot_count);

    // Remap raw pointers to vectors
    vector<Ciphertext> client_share_ct(num_ct);
    istringstream is;
    uint64_t ct_size;
    memcpy(&ct_size, client_share, sizeof(uint64_t));
    for (int ct_idx = 0; ct_idx < num_ct; ct_idx++) {
        u64 idx = sizeof(uint64_t) + ct_idx*ct_size;
        is.rdbuf()->pubsetbuf(client_share + idx, ct_size);
        client_share_ct[ct_idx].unsafe_load(is);
    }

    return client_share_postprocess(a_share, b_share, client_share_ct, num_triples, *evaluator, *encoder, *decryptor);
}

/* Free client's allocated keys */
void client_free_keys(const ClientFHE* cfhe) {
    // Free CFHE object
    delete (BatchEncoder*) cfhe->encoder;
    delete (Encryptor*) cfhe->encryptor;
    delete (Evaluator*) cfhe->evaluator;
    delete (Decryptor*) cfhe->decryptor;
}

/* Free the keyshare */
void free_key_share(char* key_share) {
    delete[] key_share;
}

/* Free server's allocated keys */
void server_free_keys(const ServerFHE *sfhe) {
    // Free SFHE object
    delete (BatchEncoder*) sfhe->encoder;
    delete (Encryptor*) sfhe->encryptor;
    delete (Evaluator*) sfhe->evaluator;
    delete (GaloisKeys*) sfhe->gal_keys;
    delete (RelinKeys*) sfhe->relin_keys;
    delete (Ciphertext*) sfhe->zero;
}

/* Free the client's state required for a single convolution */
void client_conv_free(const Metadata *data, char* ciphertext, uint64_t** result, Mode mode) {
    // Free ciphertext - Input mode ciphertext is a slightly different
    // structure but Output and InputConv are the same
    delete[] ciphertext;

    // Free result
    for (int idx = 0; idx < data->out_chans; idx++) {
        delete[] result[idx];
    }
    delete[] result;
}

/* Free the server's state required for a single convolution*/
void server_conv_free(const Metadata *data, char**** masks, char** enc_noise, char* enc_result, Mode mode) {
    // Free masks
    // Output has a different structure than InputConv and Input modes
    if (mode == Mode::Output) {
        for (int conv = 0; conv < data->convs; conv++) {
            for (int ct_idx = 0; ct_idx < data->inp_ct; ct_idx++) {
                for (int rot = 0; rot < data->filter_size; rot++) {
                    delete (Plaintext*) masks[conv][ct_idx][rot]; 
                }
                delete[] masks[conv][ct_idx];
            }
            delete[] masks[conv];
        } 
    } else {
        for (int ct_idx = 0; ct_idx < data->inp_ct; ct_idx++) {
            for (int rot = 0; rot < data->filter_size; rot++) {
                delete (Plaintext*) masks[0][ct_idx][rot]; 
            }
            delete[] masks[0][ct_idx];
        }
        delete[] masks[0];
    }
    delete[] masks;

    // Delete noise Ciphertext    
    for (int ct = 0; ct < data->out_ct; ct++) {
        delete (Ciphertext*) enc_noise[ct];
    }
    delete[] enc_result;
    delete[] enc_noise;
}

/* Free the client's state required for a single fc layer */
void client_fc_free(char* ciphertext, uint64_t* result) {
   delete[] ciphertext;
   delete[] result;
}

/* Free the server's state required for a single fc layer*/
void server_fc_free(const Metadata* data, char** enc_matrix, char* enc_noise, char* enc_result) {
    delete (Ciphertext*) enc_noise;
    delete[] enc_result;

    for (int idx = 0; idx < data->inp_ct; idx++) {
        delete (Plaintext*) enc_matrix[idx];
    }
    delete[] enc_matrix;
}

/* Free a ciphertext message passed for triple generation */
void triples_free(char* ciphertext) {
    delete[] ciphertext;
}
