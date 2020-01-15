/*
 *  Benchmark homomorphic convolution
 *
 *  Created on: August 10, 2019
 *      Author: ryanleh
 */
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <memory>
#include <limits>

#include <math.h>

#include "conv2d.h"
#include "fc_layer.h"
#include "im2col.h"

using namespace std;

/* Generates a random image and filters with the given dimensions and times
 * convolution operation */
void benchmark(int image_h, int image_w, int filter_h, int filter_w, 
        int inp_chans, int out_chans, int stride, bool padding_valid) {
    
    chrono::high_resolution_clock::time_point time_start, time_end;
    
    // Create uniform distribution
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(0, 1<<10);

    // Create Eigen inputs for the plaintext and raw arrays for HE
    EImage eimage(inp_chans);
    Image image = new Channel[inp_chans];
    for (int chan = 0; chan < inp_chans; chan++) {
        EChannel tmp_chan(image_h, image_w);
        image[chan] = new u64[image_h*image_w];
        for (int idx = 0; idx < image_h*image_w; idx++) {
            u64 val = dis(gen);
            tmp_chan(idx/image_w, idx%image_w) = val;
            image[chan][idx] = val;
        }
        eimage[chan] = tmp_chan;
    }

    EFilters efilters(out_chans);
    Filters filters = new Image[out_chans];
    for (int out_c = 0; out_c < out_chans; out_c++) {
        EImage tmp_img(inp_chans);
        filters[out_c] = new Channel[inp_chans];
        for (int inp_c = 0; inp_c < inp_chans; inp_c++) {
            EChannel tmp_chan(filter_h, filter_w);
            filters[out_c][inp_c] = new u64[filter_h*filter_w];
            for (int idx = 0; idx < filter_h*filter_w; idx++) {
                u64 val = dis(gen);
                tmp_chan(idx/filter_w, idx%filter_w) = val;
                filters[out_c][inp_c][idx] = val;
            }
            tmp_img[inp_c] = tmp_chan;
        }
        efilters[out_c] = tmp_img;
    }
    
    cout << "\n\n--------------------------------------------\n";
    cout << "Image shape: (" << image_h << "x" << image_w << ", " << inp_chans
        << ") - Filters shape: (" << filter_h << "x" << filter_w << ", " << out_chans
        << ") - Padding = " << (padding_valid ? "VALID" : "SAME") << ", Stride = (" <<
        stride << "x" << stride << ")\n";
    cout << "--------------------------------------------\n\n";

    cout << "Plaintext run ... ";
    time_start = chrono::high_resolution_clock::now();
    auto result = im2col_conv2D(&eimage, &efilters, padding_valid, stride, stride);
    time_end = chrono::high_resolution_clock::now();
    auto time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "Done [" << time_diff.count() << " microseconds]\n\n";

    int output_h = result[0].rows();
    int output_w = result[0].cols();

    cout << "Homomorphic run ... \n\n";
    time_start = chrono::high_resolution_clock::now();
    auto enc_result = HE_packed(image, filters, image_h, image_w, filter_h, filter_w, inp_chans,
            out_chans, padding_valid, stride, stride, Mode::Output);
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "Done [" << time_diff.count() << " microseconds]\n\n";

    //print(enc_result, out_chans, output_h, output_w);
    bool pass = true;
    for (int i = 0; i < out_chans; i++) {
        for (int j = 0; j < output_h * output_w; j++) {
            if (result[i](j/output_w,j%output_w) != enc_result[i][j])
                pass = false;
        }
    }
    if (pass)
        cout << "EQUAL\n" << endl;
    else
        cout << "FAIL\n" << endl;
}

int main()
{
    /* Debugging */
    //benchmark(3, 3, 2, 2, 1, 1, 1, 0);
    //benchmark(32, 32, 3, 3, 16, 16, 1, 0);
    //benchmark(16, 16, 3, 3, 32, 32, 1, 0);
    //benchmark(8, 8, 3, 3, 64, 64, 1, 1);
    
    /* Gazelle Benchmarks */
    //benchmark(28, 28, 5, 5, 5, 5, 1, 1);
    //benchmark(16, 16, 1, 1, 128, 128, 1, 1);
    //benchmark(32, 32, 3, 3, 32, 32, 1, 1);
    //benchmark(16, 16, 3, 3, 128, 128, 1, 1);

    /* ResNet Benchmarks */
    //benchmark(32, 32, 3, 3, 3, 16, 1, 0);
    //benchmark(32, 32, 3, 3, 16, 16, 1, 0);
    //benchmark(32, 32, 1, 1, 16, 16, 1, 1);
    //benchmark(32, 32, 3, 3, 16, 32, 2, 0);
    //benchmark(16, 16, 3, 3, 32, 32, 1, 1);
    //benchmark(16, 16, 3, 3, 32, 64, 2, 0);
    //benchmark(8, 8, 3, 3, 64, 64, 1, 1);

    /* Minionn Benchmarks */
    //benchmark(32, 32, 3, 3, 3, 64, 1, 0);
    //benchmark(32, 32, 3, 3, 64, 64, 1, 0);
    //benchmark(16, 16, 3, 3, 64, 64, 1, 0);
    //benchmark(8, 8, 1, 1, 64, 64, 1, 1);
    //benchmark(8, 8, 1, 1, 64, 16, 1, 1);
    return 0;
}
