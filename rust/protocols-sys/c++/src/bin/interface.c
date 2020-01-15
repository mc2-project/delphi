/*
 *  Example of using Delphi Offline's C interface
 *
 *  Created on: June 10, 2019
 *      Author: ryanleh
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "interface.h"
#include <time.h>

typedef uint64_t u64;

void conv(ClientFHE* cfhe, ServerFHE* sfhe, int image_h, int image_w, int filter_h, int filter_w,
    int inp_chans, int out_chans, int stride, bool pad_valid, Mode mode) {

  Metadata data = conv_metadata(cfhe->encoder, image_h, image_w, filter_h, filter_w, inp_chans, 
      out_chans, stride, stride, pad_valid, mode);
 
  printf("\nClient Preprocessing: ");
  float origin = (float)clock()/CLOCKS_PER_SEC;

  u64** input = (u64**) malloc(sizeof(u64*)*data.inp_chans);
  for (int chan = 0; chan < data.inp_chans; chan++) {
      input[chan] = (u64*) malloc(sizeof(u64)*data.image_size);
      for (int idx = 0; idx < data.image_size; idx++)
          input[chan][idx] = 2;
  }

  uint64_t enc_size;
  char* ciphertext;
  ciphertext = client_conv_preprocess(input, cfhe, &data, mode, &enc_size);

  float endTime = (float)clock()/CLOCKS_PER_SEC;
  float timeElapsed = endTime - origin;
  printf("[%f seconds]\n", timeElapsed);


  printf("Server Preprocessing: ");
  float startTime = (float)clock()/CLOCKS_PER_SEC;

  u64*** filters = (u64***) malloc(sizeof(u64**)*data.out_chans);
  for (int out_c = 0; out_c < data.out_chans; out_c++) {
      filters[out_c] = (u64**) malloc(sizeof(u64*)*data.inp_chans);
      for (int inp_c = 0; inp_c < data.inp_chans; inp_c++) {
          filters[out_c][inp_c] = (u64*) malloc(sizeof(u64)*data.filter_size);
          for (int idx = 0; idx < data.filter_size; idx++)
              filters[out_c][inp_c][idx] = 1;
      }
  }

  uint64_t** secret_share = (uint64_t**) malloc(sizeof(uint64_t*)*data.out_chans);
  for (int chan = 0; chan < data.out_chans; chan++) {
      secret_share[chan] = (uint64_t*) malloc(sizeof(uint64_t)*data.output_h*data.output_w);
      for (int idx = 0; idx < data.output_h*data.output_w; idx++)
        secret_share[chan][idx] = 2;
  }

  char**** masks = server_conv_preprocess(filters, sfhe, &data, mode); 
  char** enc_noise = conv_preprocess_noise(sfhe, &data, secret_share);

  endTime = (float)clock()/CLOCKS_PER_SEC;
  timeElapsed = endTime - startTime;
  printf("[%f seconds]\n", timeElapsed);

  printf("Convolution: ");
  startTime = (float)clock()/CLOCKS_PER_SEC;

  uint64_t result_size = 0;
  char* enc_result = server_conv_online(ciphertext, masks, sfhe, &data, mode, enc_noise, &result_size);
  
  endTime = (float)clock()/CLOCKS_PER_SEC;
  timeElapsed = endTime - startTime;
  printf("[%f seconds]\n", timeElapsed);

  printf("Post process: ");
  startTime = (float)clock()/CLOCKS_PER_SEC;

  u64** result = client_conv_decrypt(enc_result, cfhe, &data);

  endTime = (float)clock()/CLOCKS_PER_SEC;
  timeElapsed = endTime - startTime;
  printf("[%f seconds]\n", timeElapsed);

  timeElapsed = endTime - origin;
  printf("Total [%f seconds]\n\n", timeElapsed);

  for (int chan = 0; chan < data.out_chans; chan++) {
      int idx = 0;
      for (int row = 0; row < data.output_h; row++) {
          printf(" [");
          int col = 0;
          for (; col < data.output_w-1; col++) {
              printf("%d, " , result[chan][row*data.output_w + col]);
          }
          printf("%d ]\n" , result[chan][row*data.output_w + col]);
      }
      printf("\n");
  }

  // Free filters
  for (int out_c = 0; out_c < data.out_chans; out_c++) {
      for (int inp_c = 0; inp_c < data.inp_chans; inp_c++)
        free(filters[out_c][inp_c]);
    free(filters[out_c]);
  }
  free(filters);

  // Free image
  for (int chan = 0; chan < data.inp_chans; chan++) {
      free(input[chan]);
  }
  free(input);

  // Free secret share
  for (int chan = 0; chan < data.out_chans; chan++) {
      free(secret_share[chan]);
  }
  free(secret_share);

  // Free C++ allocations
  client_conv_free(&data, ciphertext, result, mode);
  server_conv_free(&data, masks, enc_noise, enc_result, mode);
}

void fc(ClientFHE* cfhe, ServerFHE* sfhe, int vector_len, int matrix_h) {
  Metadata data = fc_metadata(cfhe->encoder, vector_len, matrix_h);
 
  printf("\nClient Preprocessing: ");
  float origin = (float)clock()/CLOCKS_PER_SEC;

  u64* input = (u64*) malloc(sizeof(u64)*vector_len);
  for (int idx = 0; idx < vector_len; idx++)
      input[idx] = 2;

  uint64_t enc_size;
  char* ciphertext;
  ciphertext = client_fc_preprocess(input, cfhe, &data, &enc_size);

  float endTime = (float)clock()/CLOCKS_PER_SEC;
  float timeElapsed = endTime - origin;
  printf("[%f seconds]\n", timeElapsed);

  printf("Server Preprocessing: ");
  float startTime = (float)clock()/CLOCKS_PER_SEC;

  u64** matrix = (u64**) malloc(sizeof(u64*)*matrix_h);
  for (int ct = 0; ct < matrix_h; ct++) {
      matrix[ct] = (u64*) malloc(sizeof(u64)*vector_len);
      for (int idx = 0; idx < vector_len; idx++)
          matrix[ct][idx] = ct*vector_len + idx;
  }

  uint64_t* secret_share = (uint64_t*) malloc(sizeof(uint64_t)*matrix_h);
  for (int idx = 0; idx < matrix_h; idx++)
        secret_share[idx] = 0;
  char** enc_matrix = server_fc_preprocess(matrix, sfhe, &data); 
  char* enc_noise = fc_preprocess_noise(sfhe, &data, secret_share);
    
  endTime = (float)clock()/CLOCKS_PER_SEC;
  timeElapsed = endTime - startTime;
  printf("[%f seconds]\n", timeElapsed);

  printf("Layer: ");
  startTime = (float)clock()/CLOCKS_PER_SEC;

  uint64_t result_size = 0;
  char* enc_result = server_fc_online(ciphertext, enc_matrix, sfhe, &data, enc_noise, &result_size);

  endTime = (float)clock()/CLOCKS_PER_SEC;
  timeElapsed = endTime - startTime;
  printf("[%f seconds]\n", timeElapsed);

  printf("Post process: ");
  startTime = (float)clock()/CLOCKS_PER_SEC;

  u64* result = client_fc_decrypt(enc_result, cfhe, &data);

  endTime = (float)clock()/CLOCKS_PER_SEC;
  timeElapsed = endTime - startTime;
  printf("[%f seconds]\n", timeElapsed);

  timeElapsed = endTime - origin;
  printf("Total [%f seconds]\n\n", timeElapsed);

  printf(" [");
  for (int idx = 0; idx < matrix_h; idx++) {
      printf("%d, " , result[idx]);
  }
  printf("] \n");

  // Free matrix
  for (int row = 0; row < matrix_h; row++)
    free(matrix[row]);
  free(matrix);

  // Free vector
  free(input);

  // Free secret share
  free(secret_share);

  // Free C++ allocations
  client_fc_free(ciphertext, result);
  server_fc_free(&data, enc_matrix, enc_noise, enc_result);
}


void beavers_triples(ClientFHE* cfhe, ServerFHE* sfhe, int num_triples) {
  printf("\nClient Preprocessing: ");
  float origin = (float)clock()/CLOCKS_PER_SEC;

  // Generate and encrypt client's shares of a and b
  u64* client_a = (u64*) malloc(sizeof(u64)*num_triples);
  u64* client_b = (u64*) malloc(sizeof(u64)*num_triples);
  for (int idx = 0; idx < num_triples; idx++) {
      client_a[idx] = 2;
      client_b[idx] = 2;
  }
 
  uint64_t enc_size;
  char* a_ct = triples_preprocess(client_a, cfhe->encoder, cfhe->encryptor, num_triples, &enc_size);
  char* b_ct = triples_preprocess(client_b, cfhe->encoder, cfhe->encryptor, num_triples, &enc_size);

  float endTime = (float)clock()/CLOCKS_PER_SEC;
  float timeElapsed = endTime - origin;
  printf("[%f seconds]\n", timeElapsed);

  printf("Server Preprocessing: ");
  float startTime = (float)clock()/CLOCKS_PER_SEC;

  // Generate server's shares of a b c
  u64* server_a = (u64*) malloc(sizeof(u64)*num_triples);
  u64* server_b = (u64*) malloc(sizeof(u64)*num_triples);
  u64* server_r = (u64*) malloc(sizeof(u64)*num_triples);
  for (int idx = 0; idx < num_triples; idx++) {
      server_a[idx] = 3;
      server_b[idx] = 3;
      server_r[idx] = 1;
  }

  endTime = (float)clock()/CLOCKS_PER_SEC;
  timeElapsed = endTime - startTime;
  printf("[%f seconds]\n", timeElapsed);

  printf("Online: ");
  startTime = (float)clock()/CLOCKS_PER_SEC;

  char* client_share_ct = server_triples_online(a_ct, b_ct, server_a, server_b, server_r, sfhe, num_triples, &enc_size);

  endTime = (float)clock()/CLOCKS_PER_SEC;
  timeElapsed = endTime - startTime;
  printf("[%f seconds]\n", timeElapsed);
  
  printf("Post process: ");
  startTime = (float)clock()/CLOCKS_PER_SEC;
  u64* client_share = client_triples_decrypt(client_a, client_b, client_share_ct, cfhe, num_triples);

  endTime = (float)clock()/CLOCKS_PER_SEC;
  timeElapsed = endTime - startTime;
  printf("[%f seconds]\n", timeElapsed);

  timeElapsed = endTime - origin;
  printf("Total [%f seconds]\n\n", timeElapsed);

  // Free allocations
  free(client_a);
  free(server_a);
  free(client_b);
  free(server_b);
  free(server_r);
  triples_free(a_ct);
  triples_free(b_ct);
  triples_free((char*) client_share); 
  triples_free(client_share_ct);
}


int main(int argc, char* argv[]) {
  char *key_share;
  uint64_t key_share_len;

  printf("Client Keygen: ");
  float startTime = (float)clock()/CLOCKS_PER_SEC;

  ClientFHE cfhe = client_keygen(&key_share, &key_share_len);

  float endTime = (float)clock()/CLOCKS_PER_SEC;
  float timeElapsed = endTime - startTime;
  printf("[%f seconds]\n", timeElapsed);
 
  printf("Server Keygen: ");
  startTime = (float)clock()/CLOCKS_PER_SEC;
  
  ServerFHE sfhe = server_keygen(key_share); 

  endTime = (float)clock()/CLOCKS_PER_SEC;
  timeElapsed = endTime - startTime;
  printf("[%f seconds]\n", timeElapsed);

  //conv(&cfhe, &sfhe, 5, 5, 3, 3, 2, 2, 1, 0, Output);
  //conv(&cfhe, &sfhe, 32, 32, 3, 3, 16, 16, 1, 0, Output);
  //conv(&cfhe, &sfhe, 16, 16, 3, 3, 32, 32, 1, 1, Output);
  //conv(&cfhe, &sfhe, 8, 8, 3, 3, 64, 64, 1, 1, Output);
  
  //fc(&cfhe, &sfhe, 6, 3);
  
  beavers_triples(&cfhe, &sfhe, 25000);
 
  client_free_keys(&cfhe);
  free_key_share(key_share);
  server_free_keys(&sfhe);

  return 1;
}
