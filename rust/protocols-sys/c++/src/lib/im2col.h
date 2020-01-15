/*
 *  Plaintext and homomorphic implementations of convolution using im2col method
 *
 *  Created on: June 10, 2019
 *      Author: ryanleh
 */
#ifndef im2col
#define im2col

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// This is to keep compatibility for im2col implementations
typedef Matrix<uint64_t, Dynamic, Dynamic, RowMajor> EChannel;
typedef vector<EChannel> EImage;
typedef vector<EImage> EFilters;

typedef vector<uint64_t> uv64;

/* Formatted printing for EFilters/EImage/EChannel/Kernel */
template <class T> void print_image(T *data);

/* Pads images according to Tensorflow specifications */
tuple<EImage, int, int> pad_image(EImage *image, bool pad_valid, int filter_h,
        int filter_w, int stride_h, int stride_w);

/* Translates image into matrix where columns are convolution kernels */
void i2c(EImage *image, EChannel *column, const int filter_h, const int filter_w, 
        const int stride_h, const int stride_w, const int output_h, const int output_w);

/* Perform convolution on given EImage with EFilters using im2col technique. */
EImage im2col_conv2D(EImage *image, EFilters *filters, bool pad_valid, int stride_h, int stride_w);

/* Perform convolution on given EImage with EFilters using im2col technique.
 * Naively batches a single column into each ciphertext. */
EImage im2col_HE_naive(EImage *image, EFilters *filters, bool pad_valid, int stride_h, int stride_w);

/* Perform convolution on given EImage with EFilters using im2col technique.
 * Batches the max number of columns into each ciphertext. */
EImage im2col_HE_IP(EImage *image, EFilters *filters, bool pad_valid, int stride_h, int stride_w);

#endif
