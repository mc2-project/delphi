/*
 *  Simple unittests for Delphi Offline C++
 *
 *  Created on: June 10, 2019
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

#include "math.h"
#include "conv2d.h"
#include "run_conv.cpp"

using namespace std;

/* Runs convolution with provided parameters on multiple conv configurations */
void test_conv(Image image, Filters filters, int image_h, int image_w, int filter_h,
        int filter_w, int inp_chans, int out_chans, bool verbose) {
 
    cout << "\n\n--------------------------------------------\n";
    cout << "Input: (" << image_h << ", " << image_w << ", " << inp_chans
        << "), Output: (" << filter_h << ", " << filter_w << ", " <<
        out_chans << ")\n";
    cout << "--------------------------------------------\n\n";

    // Remove if you want verbose debugging
    if (!verbose)
      cout.setstate(std::ios_base::failbit);

    bool pass = true;

    cout << "\n\n--------------------------------------------\n";
    cout << "Config 1 - padding = VALID, stride = (1, 1)\n";
    cout << "--------------------------------------------\n\n";
    pass = pass & run_conv(image, filters, image_h, image_w, filter_h, filter_w,
            inp_chans, out_chans, 1, 1, 1, verbose);

    cout << "\n\n--------------------------------------------\n";
    cout << "Config 2 - padding = VALID, stride = (2, 2)\n";
    cout << "--------------------------------------------\n\n";
    pass = pass & run_conv(image, filters, image_h, image_w, filter_h, filter_w,
            inp_chans, out_chans, 1, 2, 2, verbose);

    cout << "\n\n--------------------------------------------\n";
    cout << "Config 3 - padding = SAME, stride = (1, 1)\n";
    cout << "--------------------------------------------\n\n";
    pass = pass & run_conv(image, filters, image_h, image_w, filter_h, filter_w,
            inp_chans, out_chans, 0, 1, 1, verbose);

    cout.clear();
    cout << "Passing all: " << ((pass) ? "True" : "False") << endl;
}

void conv() {
    // Example 1
    u64 image0_c1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    u64 image0_c2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    
    Channel image0[] = { image0_c1, image0_c2 };

    u64 ker0_c1[] = {1, 2, 3, 4};
    u64 ker0_c2[] = {1, 1, 1, 1};

    Channel kernel0[] = { ker0_c1, ker0_c2 };

    Image filters0[] = { kernel0 };

    test_conv(image0, filters0, 3, 3, 2, 2, 2, 1, false);
 
    // Example 2 - multiple filters
    u64 image2_c1[] = {0, 0, 0, 0, 1, 
                       1, 2, 0, 1, 0,
                       0, 1, 2, 0, 0, 
                       1, 0, 0, 2, 1,
                       1, 2, 1, 0, 0}; 
    u64 image2_c2[] = {1, 1, 1, 1, 1, 
                       2, 2, 2, 1, 0, 
                       1, 0, 0, 0, 2,
                       0, 1, 2, 1, 0,
                       1, 1, 2, 0, 1};
    u64 image2_c3[] = {0, 0, 2, 2, 2,
                       1, 2, 0, 1, 2,
                       1, 0, 2, 2, 1,
                       0, 2, 0, 1, 0,
                       2, 1, 1, 1, 0};
    Channel image2[] = { image2_c1, image2_c2, image2_c3 };

    u64 ker1_c1[] = {1, 1, 0,
                     0, 0, 0, 
                     1, 0, 0};
    u64 ker1_c2[] = {1, 1, 1,
                     1, 0, 1,
                     1, 1, 1};
    u64 ker1_c3[] = {1, 1, 1,
                     0, 1, 0, 
                     1, 1, 1};
    Channel kernel1[] = { ker1_c1, ker1_c2, ker1_c3 };

    u64 ker2_c1[] = {2, 1, 0,
                     0, 0, 0, 
                     2, 0, 0};
    u64 ker2_c2[] = {1, 2, 2,
                     2, 0, 1,
                     1, 1, 2};
    u64 ker2_c3[] = {2, 2, 1,
                     0, 1, 0, 
                     1, 2, 1};
    Channel kernel2[] = { ker2_c1, ker2_c2, ker2_c3 };
   
    u64 ker3_c1[] = {1, 1, 1,
                     1, 1, 1,
                     1, 1, 1};
    u64 ker3_c2[] = {1, 1, 1,
                     1, 1, 1,
                     1, 1, 1};
    u64 ker3_c3[] = {1, 1, 1,
                     1, 1, 1,
                     1, 1, 1};
    Channel kernel3[] =  { ker3_c1, ker3_c2, ker3_c3 };

    Image filters2[] = { kernel1, kernel2, kernel3 };

    test_conv(image2, filters2, 5, 5, 3, 3, 3, 3, false);


    // Example 3 - multiple filters but output channel size > input channels
    u64 image5_c1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5};
    u64 image5_c2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6};
    u64 image5_c3[] = {2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7};
    Channel image5[] = { image5_c1, image5_c2, image5_c3 };
    
    u64 ker8_c1[] = {0, 0, 1, 1};
    u64 ker8_c2[] = {1, 0, 0, 0};
    u64 ker8_c3[] = {1, 0, 0, 0};

    u64 ker9_c1[] = {0, 0, 2, 2};
    u64 ker9_c2[] = {2, 0, 0, 0};
    u64 ker9_c3[] = {2, 0, 0, 0};
    
    u64 ker10_c1[] = {0, 0, 3, 3};
    u64 ker10_c2[] = {3, 0, 0, 0};
    u64 ker10_c3[] = {3, 0, 0, 0};

    u64 ker11_c1[] = {0, 0, 4, 4};
    u64 ker11_c2[] = {4, 0, 0, 0};
    u64 ker11_c3[] = {4, 0, 0, 0};

    Channel kernel8[] = { ker8_c1, ker8_c2, ker8_c3 };
    Channel kernel9[] = { ker9_c1, ker9_c2, ker9_c3 };
    Channel kernel10[] = { ker10_c1, ker10_c2, ker10_c3 };
    Channel kernel11[] = { ker11_c1, ker11_c2, ker11_c3 };

    Image filters5[] = { kernel8, kernel9, kernel10, kernel11 };

    test_conv(image5, filters5, 4, 4, 2, 2, 3, 4, false);

    // Example 4 - Inp = Out but spans multiple halves
    u64 image6_c1[1024];
    u64 image6_c2[1024];
    u64 image6_c3[1024];
    u64 image6_c4[1024];
    // This channel should be in the second half of the ciphertext
    u64 image6_c5[1024];
    for (int i = 0; i < 1024; i++) {
        image6_c1[i] = 1; 
        image6_c2[i] = 1; 
        image6_c3[i] = 1; 
        image6_c4[i] = 1; 
        image6_c5[i] = 1; 
    } 
    Channel image6[] =  { image6_c1, image6_c2, image6_c3, image6_c4, image6_c5 };

    u64 ker12_c1[] = {0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0};
    u64 ker12_c2[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
    u64 ker12_c3[] = {1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1};
    u64 ker12_c4[] = {0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0};
    u64 ker12_c5[] = {0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1};

    u64 ker13_c1[] = {0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0};
    u64 ker13_c2[] = {2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0};
    u64 ker13_c3[] = {2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2};
    u64 ker13_c4[] = {0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0};
    u64 ker13_c5[] = {0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 2};

    u64 ker14_c1[] = {0, 0, 3, 3, 0, 0, 0, 3, 3, 0, 0, 0, 3, 3, 0, 0, 0, 3, 3, 0, 0, 0, 3, 3, 0};
    u64 ker14_c2[] = {3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0};
    u64 ker14_c3[] = {3, 0, 0, 0, 3, 3, 0, 0, 0, 3, 3, 0, 0, 0, 3, 3, 0, 0, 0, 3, 3, 0, 0, 0, 3};
    u64 ker14_c4[] = {0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0};
    u64 ker14_c5[] = {0, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 0, 3, 0, 3};

    u64 ker15_c1[] = {0, 0, 4, 4, 0, 0, 0, 4, 4, 0, 0, 0, 4, 4, 0, 0, 0, 4, 4, 0, 0, 0, 4, 4, 0};
    u64 ker15_c2[] = {4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0};
    u64 ker15_c3[] = {4, 0, 0, 0, 4, 4, 0, 0, 0, 4, 4, 0, 0, 0, 4, 4, 0, 0, 0, 4, 4, 0, 0, 0, 4};
    u64 ker15_c4[] = {0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0};
    u64 ker15_c5[] = {0, 0, 4, 0, 4, 0, 0, 4, 0, 4, 0, 0, 4, 0, 4, 0, 0, 4, 0, 4, 0, 0, 4, 0, 4};

    u64 ker16_c1[] = {0, 0, 5, 5, 0, 0, 0, 5, 5, 0, 0, 0, 5, 5, 0, 0, 0, 5, 5, 0, 0, 0, 5, 5, 0};
    u64 ker16_c2[] = {5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0};
    u64 ker16_c3[] = {5, 0, 0, 0, 5, 5, 0, 0, 0, 5, 5, 0, 0, 0, 5, 5, 0, 0, 0, 5, 5, 0, 0, 0, 5};
    u64 ker16_c4[] = {0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0};
    u64 ker16_c5[] = {0, 0, 5, 0, 5, 0, 0, 5, 0, 5, 0, 0, 5, 0, 5, 0, 0, 5, 0, 5, 0, 0, 5, 0, 5};

    Channel kernel12[] = { ker12_c1, ker12_c2, ker12_c3, ker12_c4, ker12_c5 };
    Channel kernel13[] = { ker13_c1, ker13_c2, ker13_c3, ker13_c4, ker13_c5 };
    Channel kernel14[] = { ker14_c1, ker14_c2, ker14_c3, ker14_c4, ker14_c5 };
    Channel kernel15[] = { ker15_c1, ker15_c2, ker15_c3, ker15_c4, ker15_c5 };
    Channel kernel16[] = { ker16_c1, ker16_c2, ker16_c3, ker16_c4, ker16_c5 };

    Image filters6[] = { kernel12, kernel13, kernel14, kernel15, kernel16 };

    test_conv(image6, filters6, 32, 32, 5, 5, 5, 5, false);

    // Example 5 - Inp < Out but spans multiple halves
    u64 ker17_c1[] = {0, 0, 6, 6, 0, 0, 0, 6, 6, 0, 0, 0, 6, 6, 0, 0, 0, 6, 6, 0, 0, 0, 6, 6, 0};
    u64 ker17_c2[] = {6, 0, 0, 0, 0, 6, 0, 0, 0, 0, 6, 0, 0, 0, 0, 6, 0, 0, 0, 0, 6, 0, 0, 0, 0};
    u64 ker17_c3[] = {6, 0, 0, 0, 6, 6, 0, 0, 0, 6, 6, 0, 0, 0, 6, 6, 0, 0, 0, 6, 6, 0, 0, 0, 6};
    u64 ker17_c4[] = {0, 6, 0, 0, 0, 0, 6, 0, 0, 0, 0, 6, 0, 0, 0, 0, 6, 0, 0, 0, 0, 6, 0, 0, 0};
    u64 ker17_c5[] = {0, 0, 6, 0, 6, 0, 0, 6, 0, 6, 0, 0, 6, 0, 6, 0, 0, 6, 0, 6, 0, 0, 6, 0, 6};
    
    u64 ker18_c1[] = {0, 0, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 7, 7, 0};
    u64 ker18_c2[] = {7, 0, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 0};
    u64 ker18_c3[] = {7, 0, 0, 0, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 7};
    u64 ker18_c4[] = {0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0};
    u64 ker18_c5[] = {0, 0, 7, 0, 7, 0, 0, 7, 0, 7, 0, 0, 7, 0, 7, 0, 0, 7, 0, 7, 0, 0, 7, 0, 7};

    Channel kernel17[] = { ker17_c1, ker17_c2, ker17_c3, ker17_c4, ker17_c5 };
    Channel kernel18[] = { ker18_c1, ker18_c2, ker18_c3, ker18_c4, ker18_c5 };

    Image filters8[] = {kernel12, kernel13, kernel14, kernel15, kernel16, kernel17, kernel18 };

    test_conv(image6, filters8, 32, 32, 5, 5, 5, 7, false);
    

    // Example 6 - Inp = Out but spans multiple ciphertexts
    u64 image7_c6[1024];
    u64 image7_c7[1024];
    u64 image7_c8[1024];
    u64 image7_c9[1024];
    for (int i = 0; i < 1024; i++) {
        image7_c6[i] = 1; 
        image7_c7[i] = 1; 
        image7_c8[i] = 1; 
        image7_c9[i] = 1; 
    } 

    u64 ker19_c1[] = {0, 0, 8, 8, 0, 0, 0, 8, 8, 0, 0, 0, 8, 8, 0, 0, 0, 8, 8, 0, 0, 0, 8, 8, 0};
    u64 ker19_c2[] = {8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0};
    u64 ker19_c3[] = {8, 0, 0, 0, 8, 8, 0, 0, 0, 8, 8, 0, 0, 0, 8, 8, 0, 0, 0, 8, 8, 0, 0, 0, 8};
    u64 ker19_c4[] = {0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0};
    u64 ker19_c5[] = {0, 0, 8, 0, 8, 0, 0, 8, 0, 8, 0, 0, 8, 0, 8, 0, 0, 8, 0, 8, 0, 0, 8, 0, 8};
    
    u64 ker20_c1[] = {0, 0, 9, 9, 0, 0, 0, 9, 9, 0, 0, 0, 9, 9, 0, 0, 0, 9, 9, 0, 0, 0, 9, 9, 0};
    u64 ker20_c2[] = {9, 0, 0, 0, 0, 9, 0, 0, 0, 0, 9, 0, 0, 0, 0, 9, 0, 0, 0, 0, 9, 0, 0, 0, 0};
    u64 ker20_c3[] = {9, 0, 0, 0, 9, 9, 0, 0, 0, 9, 9, 0, 0, 0, 9, 9, 0, 0, 0, 9, 9, 0, 0, 0, 9};
    u64 ker20_c4[] = {0, 9, 0, 0, 0, 0, 9, 0, 0, 0, 0, 9, 0, 0, 0, 0, 9, 0, 0, 0, 0, 9, 0, 0, 0};
    u64 ker20_c5[] = {0, 0, 9, 0, 9, 0, 0, 9, 0, 9, 0, 0, 9, 0, 9, 0, 0, 9, 0, 9, 0, 0, 9, 0, 9};

    Channel image7[] = { image6_c1, image6_c2, image6_c3, image6_c4, image6_c5, image7_c6, image7_c7, image7_c8, image7_c9 };

    Channel kernel19[] = { ker12_c1, ker12_c2, ker12_c3, ker12_c4, ker12_c5, ker12_c1, ker12_c2, ker12_c3, ker12_c4 };
    Channel kernel20[] = { ker13_c1, ker13_c2, ker13_c3, ker13_c4, ker13_c5, ker13_c1, ker13_c2, ker13_c3, ker13_c4 };
    Channel kernel21[] = { ker14_c1, ker14_c2, ker14_c3, ker14_c4, ker14_c5, ker14_c1, ker14_c2, ker14_c3, ker14_c4 };
    Channel kernel22[] = { ker15_c1, ker15_c2, ker15_c3, ker15_c4, ker15_c5, ker15_c1, ker15_c2, ker15_c3, ker15_c4 };
    Channel kernel23[] = { ker16_c1, ker16_c2, ker16_c3, ker16_c4, ker16_c5, ker16_c1, ker16_c2, ker16_c3, ker16_c4 };
    Channel kernel24[] = { ker17_c1, ker17_c2, ker17_c3, ker17_c4, ker17_c5, ker17_c1, ker17_c2, ker17_c3, ker17_c4 };
    Channel kernel25[] = { ker18_c1, ker18_c2, ker18_c3, ker18_c4, ker18_c5, ker18_c1, ker18_c2, ker18_c3, ker18_c4 };
    Channel kernel26[] = { ker19_c1, ker19_c2, ker19_c3, ker19_c4, ker19_c5, ker19_c1, ker19_c2, ker19_c3, ker19_c4 };
    Channel kernel27[] = { ker20_c1, ker20_c2, ker20_c3, ker20_c4, ker20_c5, ker20_c1, ker20_c2, ker20_c3, ker20_c4 };

    Image filters9[] = { kernel19, kernel20, kernel21, kernel22, kernel23, kernel24, kernel25, kernel26, kernel27 };

    test_conv(image7, filters9, 32, 32, 5, 5, 9, 9, false);

    // Example 7 - Inp < Out but spans multiple ciphertexts
    Image filters10[] = {kernel19, kernel20, kernel21, kernel22, kernel23, kernel24, kernel25, kernel26, kernel27, kernel26 };

    test_conv(image7, filters10, 32, 32, 5, 5, 9, 10, false);

    // Example 8 - Inp = Out but spans multiple ciphertexts and halves
    Channel image8[] = { image6_c1, image6_c2, image6_c3, image6_c4, image6_c5, image7_c6, image7_c7, image7_c8, image7_c9,
                         image7_c8, image7_c7, image7_c6, image6_c5, image6_c4, image6_c3 };

    Channel kernel28[] = { ker12_c1, ker12_c2, ker12_c3, ker12_c4, ker12_c5, ker12_c1, ker12_c2, ker12_c3, ker12_c4,
                     ker12_c5, ker12_c1, ker12_c2, ker12_c3, ker12_c4, ker12_c5};

    Channel kernel29[] = { ker13_c1, ker13_c2, ker13_c3, ker13_c4, ker13_c5, ker13_c1, ker13_c2, ker13_c3, ker13_c4,
                     ker13_c5, ker13_c1, ker13_c2, ker13_c3, ker13_c4, ker13_c5};

    Channel kernel30[] = { ker14_c1, ker14_c2, ker14_c3, ker14_c4, ker14_c5, ker14_c1, ker14_c2, ker14_c3, ker14_c4,
                     ker14_c5, ker14_c1, ker14_c2, ker14_c3, ker14_c4, ker14_c5};

    Channel kernel31[] = { ker15_c1, ker15_c2, ker15_c3, ker15_c4, ker15_c5, ker15_c1, ker15_c2, ker15_c3, ker15_c4,
                     ker15_c5, ker15_c1, ker15_c2, ker15_c3, ker15_c4, ker15_c5};

    Channel kernel32[] = { ker16_c1, ker16_c2, ker16_c3, ker16_c4, ker16_c5, ker16_c1, ker16_c2, ker16_c3, ker16_c4,
                     ker16_c5, ker16_c1, ker16_c2, ker16_c3, ker16_c4, ker16_c5};

    Channel kernel33[] = { ker17_c1, ker17_c2, ker17_c3, ker17_c4, ker17_c5, ker17_c1, ker17_c2, ker17_c3, ker17_c4,
                     ker17_c5, ker17_c1, ker17_c2, ker17_c3, ker17_c4, ker17_c5};

    Channel kernel34[] = { ker18_c1, ker18_c2, ker18_c3, ker18_c4, ker18_c5, ker18_c1, ker18_c2, ker18_c3, ker18_c4,
                     ker18_c5, ker18_c1, ker18_c2, ker18_c3, ker18_c4, ker18_c5};

    Channel kernel35[] = { ker19_c1, ker19_c2, ker19_c3, ker19_c4, ker19_c5, ker19_c1, ker19_c2, ker19_c3, ker19_c4,
                     ker19_c5, ker19_c1, ker19_c2, ker19_c3, ker19_c4, ker19_c5};

    Channel kernel36[] = { ker20_c1, ker20_c2, ker20_c3, ker20_c4, ker20_c5, ker20_c1, ker20_c2, ker20_c3, ker20_c4,
                     ker20_c5, ker20_c1, ker20_c2, ker20_c3, ker20_c4, ker20_c5};

    Image filters11[] = { kernel28, kernel29, kernel30, kernel31, kernel32, kernel33, kernel34, kernel35, kernel36, kernel35,
                         kernel34, kernel33, kernel32, kernel31, kernel30};

    test_conv(image8, filters11, 32, 32, 5, 5, 15, 15, false);


    // Example 9 - Inp < Out but spans multiple ciphertexts and halves
    Image filters12[] = {kernel28, kernel29, kernel30, kernel31, kernel32, kernel33, kernel34, kernel35, kernel36, kernel35,
                         kernel34, kernel33, kernel32, kernel31, kernel30, kernel29, kernel28};

    test_conv(image8, filters12, 32, 32, 5, 5, 15, 17, false);
}

int main()
{
    conv();
    return 0;
}
