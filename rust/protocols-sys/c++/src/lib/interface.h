/*
 *  C interface for rust interopt
 *
 *  Created on: August 10, 2019
 *      Author: ryanleh
 */
#ifndef interface
#define interface

#include <stdint.h>
#include <stdbool.h>

const uint64_t PLAINTEXT_MODULUS = 2061584302081;
const uint64_t POLY_MOD_DEGREE = 8192;    

/*
 * Each type here refers to how much preprocessing the client is doing in order
 * of bandwidth cost
 *   InputConv   -> user does input packing + filter rotations + convolution rotations
 *   Input       -> user does input packing + filter rotations
 *   Output      -> user does output packing + filter rotations
 */
typedef enum Mode {
    InputConv,
    Input,
    Output,
} Mode;

/* This is ugly but very useful. */
typedef struct Metadata {
    /* Number of plaintext slots in a ciphertext */
    int32_t slot_count;
    /* Number of plaintext slots in a half ciphertext (since ciphertexts are a 
     * two column matrix) */
    int32_t pack_num;
    /* Number of Channels that can fit in a half ciphertext */
    int32_t chans_per_half;
    /* Number of input ciphertexts for convolution (this may not match what the
     * client actually transmits) */
    int32_t inp_ct;
    /* Number of output ciphertexts */
    int32_t out_ct;
    /* Image and Filters metadata */
    int32_t image_h;
    int32_t image_w;
    int32_t image_size;
    int32_t inp_chans;
    int32_t filter_h;
    int32_t filter_w;
    int32_t filter_size;
    int32_t out_chans;
    /* How many total ciphertext halves the input and output take up */
    int32_t inp_halves;
    int32_t out_halves;
    /* How many Channels are in the last output or input half */
    int32_t out_in_last;
    int32_t inp_in_last;
    /* The modulo used when deciding which output channels to pack into a mask */
    int32_t out_mod;
    /* How many permutations of ciphertexts are needed to generate all
     * int32_termediate rotation sets (Output Packing Only) */
    int32_t half_perms;
    /* Whether the last output half can fit repeats (Output Packing Only) */
    bool last_repeats;
    /* The number of Channels in a single repeat. This may require padding
     * since the repeat must be a power of 2 (Output Packing Only) */
    int32_t repeat_chans;
    /* The number of rotations for each ciphertext half (Output Packing Only) */
    int32_t half_rots;
    /* The number of rotations for the last ciphertext half (Output Packing Only) */
    int32_t last_rots;
    /* Total number of convolutions needed to generate all intermediate
     * rotations sets (Output Packing Only) */
    int32_t convs;
    int32_t stride_h;
    int32_t stride_w;
    int32_t output_h;
    int32_t output_w;
    int32_t pad_t;
    int32_t pad_b;
    int32_t pad_r;
    int32_t pad_l;
} Metadata;

#ifdef __cplusplus
extern "C" {
#endif
    typedef struct ClientFHE {
        void* encoder;
        void* encryptor;
        void* evaluator;
        void* decryptor;
    } ClientFHE;

    typedef struct ServerFHE {
        void* encoder;
        void* encryptor;
        void* evaluator;
        void* gal_keys;
        void* relin_keys;
        char* zero;
    } ServerFHE;
    
    /* Generates new keys and helpers for Client. Returns the helpers and allocates
     * a bytestream for sharing the keys with the server */
    ClientFHE client_keygen(char** key_share, uint64_t* key_share_len);

    /* Generates new helpers for Server given a public_key */
    ServerFHE server_keygen(char* key_share);

    /* Populates the Metadata struct for Conv layer */
    Metadata conv_metadata(void* batch_encoder, int32_t image_h, int32_t image_w,
            int32_t filter_h, int32_t filter_w, int32_t inp_chans, int32_t out_chans, int32_t stride_h,
            int32_t stride_w, bool pad_valid, Mode mode);

    /* Populates the Metadata struct for FC layer */
    Metadata fc_metadata(void* batch_encoder, int32_t vector_len, int32_t matrix_h);
    
    /* Allocates correct serialized objects for each provided pointer */
    char* client_conv_preprocess(const uint64_t* const* image, const ClientFHE* cfhe, const Metadata* data, Mode mode, uint64_t* enc_size);
    
    /* Generates necessary filters and stores in tensor */
    char**** server_conv_preprocess(const uint64_t* const* const* filters, const ServerFHE* sfhe, const Metadata* data, Mode mode);

    /* Preprocess and serializes client input vector */
    char* client_fc_preprocess(const uint64_t* vector, const ClientFHE* cfhe, const Metadata* data, uint64_t* enc_size);

    /* Preprocesses server matrix and enc_noise */
    char** server_fc_preprocess(const uint64_t* const* matrix, const ServerFHE* sfhe, const Metadata* data);

    /* Encrypts client's a for each multiplication triple */
    char* triples_preprocess(const uint64_t* share, void* encoder_c, void* encryptor_c, int num_triples, uint64_t* enc_size);

    /* Computes the masking noise ciphertext for conv output */        
    char** conv_preprocess_noise(const ServerFHE* sfhe, const Metadata* data, const uint64_t* const* secret_share);
    
    /* Computes the masking noise ciphertext for conv output */
    char* fc_preprocess_noise(const ServerFHE* sfhe, const Metadata* data, const uint64_t* secret_share);

    /* Performs the convolution on the given input */
    char* server_conv_online(char* ciphertext, char**** masks, const ServerFHE* sfhe, const Metadata* data,
            Mode mode, char** enc_noise, uint64_t* enc_result_size);
    
    /* Performs matrix multiplication on the given inputs */
    char* server_fc_online(char* ciphertext, char** enc_matrix, const ServerFHE* sfhe, const Metadata* data,
            char* enc_noise, uint64_t* enc_result_size);

    /* Computes the encrypted client's share of multiplication triple */
    char* server_triples_online(char* client_a, char* client_b, const uint64_t* a_share,
        const uint64_t* b_share, const uint64_t* r_share, const ServerFHE* sfhe,
        int num_triples, uint64_t* enc_size);

    /* Decrypts and reshapes convolution result */
    uint64_t** client_conv_decrypt(char* enc_result, const ClientFHE *cfhe, const Metadata *data);
    
    /* Decrypts and reshapes fully-connected result */
    uint64_t* client_fc_decrypt(char* enc_result, const ClientFHE *cfhe, const Metadata *data);

    /* Decrypts the clients multiplication triple share */
    uint64_t* client_triples_decrypt(const uint64_t* a_share, const uint64_t* b_share, char* client_share, const ClientFHE* cfhe, int num_triples);

    /* Free client's allocated keys */
    void client_free_keys(const ClientFHE* cfhe); 
 
    /* Free server's allocated keys */
    void server_free_keys(const ServerFHE *sfhe);
    
    /* Free the keyshare */
    void free_key_share(char* key_share); 

    /* Free the client's state required for a single convolution */
    void client_conv_free(const Metadata* data, char* ciphertext, uint64_t** result, Mode mode);
   
    /* Free the server's state required for a single convolution*/
    void server_conv_free(const Metadata* data, char**** masks, char** enc_noise, char* enc_result, Mode mode);
    
    /* Free the client's state required for a single fc layer */
    void client_fc_free(char* ciphertext, uint64_t* result);

    /* Free the server's state required for a single fc layer*/
    void server_fc_free(const Metadata* data, char** enc_matrix, char* enc_noise, char* enc_result);

    /* Free a ciphertext message passed for triple generation */
    void triples_free(char* ciphertext);
    
#ifdef __cplusplus
}
#endif

#endif
