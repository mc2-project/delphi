#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "seal/seal.h"
#include "im2col.h"
#include "interface.h"

#include <bitset>

using namespace Eigen;
using namespace seal;
using namespace std;

template <class T>
void print_image(T *data) {

    IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", " ", " ");
    // Save the formatting information for cout.
    ios old_fmt(nullptr);
    old_fmt.copyfmt(cout);

    // Set precision to 3
    cout << fixed << setprecision(1) << endl;

    // constexpr is so compiler doesn't freak out with the generic type
    // comparison
    if constexpr (is_same<T, EFilters>::value) {
        cout << "------FILTERS------";
        for (auto &filter: *data) {
            print_image(&filter); 
        } 
        cout << "------------------" << endl;
    } else if constexpr (is_same<T, EImage>::value) {
        int height = (*data)[0].rows();

        cout << "------IMAGE------" << endl;
        for (int i = 0; i < height; i++) {
            for (auto &channel: *data) {
                cout << " [" << channel.row(i).format(CommaInitFmt) << "], ";
            }
            cout << endl;
        }
        cout << "-----------------" << endl;
    } else {
        int height = (*data).rows();
        int width = (*data).cols();

        for (int i = 0; i < height; i++) {
            cout << " [" << (*data).row(i).format(CommaInitFmt) << "], ";
            cout << endl;
        }
        cout << endl;
    }
    //Restore the old cout formatting.
    cout.copyfmt(old_fmt);
}


/* Use casting to do two conditionals instead of one - check if a > 0 and a < b */
inline bool condition_check(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b); 
}


/* Adapted im2col algorithm from Caffe framework */
void i2c(EImage *image, EChannel *column, const int filter_h, const int filter_w,
        const int stride_h, const int stride_w, const int output_h, const int output_w) {
    int height = (*image)[0].rows();
    int width = (*image)[0].cols();
    int channels = (*image).size();

    int col_width = (*column).cols();
    
    // Index counters for images
    int column_i = 0;
    const int channel_size = height * width;
    for (auto &channel: (*image)) {
        for (int filter_row = 0; filter_row < filter_h; filter_row++) {
            for (int filter_col = 0; filter_col < filter_w; filter_col++) {
                int input_row = filter_row;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (!condition_check(input_row, height)) {
                        for (int output_cols = output_w; output_cols; output_cols--) {
                            //column->data[column_i] = 0;
                            int row_i = column_i / col_width;
                            int col_i = column_i % col_width;
                            (*column)(row_i, col_i) = 0;
                            column_i++;
                        } 
                    } else {
                        int input_col = filter_col;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (condition_check(input_col, width)) {
                                //column->data[col_index] = image->data[image_i + input_row * width + input_col];
                                int row_i = column_i / col_width;
                                int col_i = column_i % col_width;
                                (*column)(row_i, col_i) = channel(input_row, input_col);
                                column_i++;
                            } else {
                                //column->data[column_i] = 0;
                                int row_i = column_i / col_width;
                                int col_i = column_i % col_width;
                                (*column)(row_i, col_i) = 0;
                                column_i++;
                            }
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}


/* Pads an image and returns the correct output_h and output_w */
std::tuple<EImage, int, int> pad_image(EImage *image, bool pad_valid, int filter_h,
        int filter_w, int stride_h, int stride_w) {
    // Calculate output matrix dimensions pad matrix. Calculations taken from
    // Tensorflow
    int output_h, output_w, pad_h, pad_w;
    // All EChannels in an EImage are of the same size
    int image_h = (*image)[0].rows();
    int image_w = (*image)[0].cols();
    EImage p_image;

    if (pad_valid) {
        output_h = ceil((float)(image_h - filter_h + 1) / stride_h);
        output_w = ceil((float)(image_w - filter_w + 1) / stride_w);
        pad_h = 0;
        pad_w = 0;
        p_image = *image;
    } else {
        output_h = ceil((float)image_h / stride_h);
        output_w = ceil((float)image_w / stride_w);
        // Total amount of vertical and horizontal padding needed
        pad_h = max((output_h - 1) * stride_h + filter_h - image_h, 0);
        pad_w = max((output_w - 1) * stride_w + filter_w - image_w, 0);
    
        // The floor is implicit with uint but put just in case type is changed
        int pad_top = floor((float)pad_h / 2);
        int pad_bottom = pad_h - pad_top;
        int pad_left = floor((float)pad_w / 2);
        int pad_right = pad_w - pad_left;
        
        // New dimensions: Height = image_h + pad_h, Width = image_w + pad_w 
        // Original image get's copied in from left corner at index: (pad_top, pad_left)
        for (EChannel &channel: *image) {
            EChannel p_channel = EChannel::Zero(image_h + pad_h, image_w + pad_w);
            p_channel.block(pad_top, pad_left, image_h, image_w) = channel;
            p_image.push_back(p_channel);
        }
    }
    return { p_image, output_h, output_w };
}


/* Perform convolution on given EImage with EFilters. DOES NOT SUPPORT DILATION */
EImage im2col_conv2D(EImage *image, EFilters *filters, bool pad_valid, int stride_h, int stride_w) {
    int channels = (*image).size();
    // All filters should have same dimensions
    int filter_h = (*filters)[0][0].rows();
    int filter_w = (*filters)[0][0].cols();

    auto [p_image, output_h, output_w] = pad_image(image, pad_valid,
                                                   filter_h, filter_w,
                                                   stride_h, stride_w);
    
    // Generate the column representation of image for convolution
    const int col_height = filter_h * filter_w * channels;
    const int col_width = output_h * output_w;
    EChannel image_col(col_height, col_width);
    i2c(&p_image, &image_col, filter_h, filter_w, stride_h, stride_w, output_h, output_w);

    // For each filter, flatten it into and multiply with image_col
    EImage result;
    for (auto &filter: *filters) {
        EChannel filter_col(1, col_height);
        // Use im2col with a filter size 1x1 to translate
        i2c(&filter, &filter_col, 1, 1, 1, 1, filter_h, filter_w); 
        EChannel tmp = filter_col * image_col;

        // Reshape result of multiplication to the right size
        // SEAL stores matrices in RowMajor form
        result.push_back(Map<Matrix<uint64_t, Dynamic, Dynamic, RowMajor>>(tmp.data(), output_h, output_w));
    }
    return result;
}


/* Perform convolution on given EImage with EFilters. DOES NOT SUPPORT DILATION */
EImage im2col_HE_naive(EImage *image, EFilters *filters, bool pad_valid, int stride_h, int stride_w) {
    int ciphertexts = 0;
    int rotations = 0;
    int multiplications = 0;
    int additions = 0;
    
    // Param and key gen
    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(8192);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(8192));
    parms.set_plain_modulus(PLAINTEXT_MODULUS);
    auto context = SEALContext(parms);
    
    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    auto secret_key = keygen.secret_key();
    // Parameter are large enough that we should be fine with these at max
    // decomposition
    GaloisKeys gal_keys;
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    keygen.create_galois_keys(gal_keys);

    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key); 
    BatchEncoder batch_encoder(context);
    size_t slot_count = batch_encoder.slot_count();

    // Pad EImage
    // All filters should have same dimensions
    int filter_h = (*filters)[0][0].rows();
    int filter_w = (*filters)[0][0].cols();
    auto [p_image, output_h, output_w] = pad_image(image, pad_valid,
                                                   filter_h, filter_w,
                                                   stride_h, stride_w);
  
    // Generate the column representation of image for convolution
    int channels = (*image).size();
    int output_channels = filters->size();
    const int col_height = filter_h * filter_w * channels;
    const int col_num = output_h * output_w;
    EChannel image_col(col_height, col_num);
    i2c(&p_image, &image_col, filter_h, filter_w, stride_h, stride_w, output_h, output_w);

    // Generate the row representation of kernel
    // For each filter, flatten it into a Kernel and encode it
    // We can keep as plaintext since server is evaluating
    vector<Plaintext> filters_encoded(output_channels);
    for (int i = 0; i < output_channels; i++) {
        EChannel filter_col(1, col_height);
        // Use im2col with a filter size 1x1 to translate
        i2c(&(*filters)[i], &filter_col, 1, 1, 1, 1, filter_h, filter_w);
        // Convert kernel to vector
        vector<uint64_t> pod_matrix(slot_count, 0ULL);
        copy(filter_col.data(), 
             filter_col.data() + col_height,
             pod_matrix.begin());
        // Encode vector
        batch_encoder.encode(pod_matrix, filters_encoded[i]);
    }

    // Encrypt each image column as a packed ciphertext
    vector<Ciphertext> image_enc(col_num);
    for (int j = 0; j < col_num; j++) {
        // Convert column to vector
        vector<uint64_t> pod_matrix(slot_count, 0ULL);
        move(image_col.col(j).data(),
                  image_col.col(j).data() + col_height,
                  pod_matrix.begin());
        // Encode and encrypt column
        Plaintext col_matrix;
        batch_encoder.encode(pod_matrix, col_matrix);
        encryptor.encrypt(col_matrix, image_enc[j]);
        ciphertexts += 1;
    }
    
    // For each kernel, perform vector-matrix multiplication
    vector<Ciphertext> results_enc(col_num * output_channels);
    for (int k = 0; k < output_channels; k++) {
        // Compute matrix multiplication for column and kernel
        // Multiply kernel and column
        vector<Ciphertext> rot_copies(col_num);
        vector<Ciphertext> mul_result(col_num);
        for (int j = 0; j < col_num; j++) {
            evaluator.multiply_plain(image_enc[j], filters_encoded[k], mul_result[j]);
            evaluator.relinearize_inplace(mul_result[j], relin_keys);
            multiplications += 1;
            // Copy each multiplication as reference for rotation
            rot_copies[j] = mul_result[j];
        }
        // Rotate and add col_height times
        for (int i = 0; i < col_height; i++) {
            for (int j = 0; j < col_num; j++) {
                evaluator.rotate_rows_inplace(rot_copies[j], -1, gal_keys);
                rotations += 1;
                evaluator.add_inplace(mul_result[j], rot_copies[j]);
                additions += 1;
            }
        }
        move(mul_result.begin(),
             mul_result.end(),
             results_enc.begin() + k*(col_num));
    }

    // Decrypt channel by channel
    EImage final_result;
    for (int i = 0; i < output_channels; i++) {
        EChannel channel_result;
        vector<uint64_t> channel_vec(col_num);
        for (int j = 0; j < col_num; j++) {
            Plaintext plain;
            vector<uint64_t> pod_matrix(slot_count, 0ULL);
            decryptor.decrypt(results_enc[i*(col_num) + j], plain);
            batch_encoder.decode(plain, pod_matrix);
            // Result is the col_height element of ciphertext
            channel_vec[j] = pod_matrix[col_height];
        }
        final_result.push_back(Map<Matrix<uint64_t, Dynamic, Dynamic, RowMajor>>(&channel_vec[0], output_h, output_w));
    }

    cout << "Ciphertexts: " << ciphertexts << endl;
    cout << "Additions: " << additions << endl;
    cout << "Rotations: " << rotations << endl;
    cout << "Multiplications: " << multiplications << endl;
    return final_result;
}

/* Perform convolution on given EImage with EFilters. DOES NOT SUPPORT DILATION */
EImage im2col_HE_IP(EImage *image, EFilters *filters, bool pad_valid, int stride_h, int stride_w) {
    chrono::high_resolution_clock::time_point time_start, time_end;
    int ciphertexts = 0;
    int rotations = 0;
    int multiplications = 0;
    int additions = 0;
    int subtractions = 0;
    
    cout << "Param and key gen";
    time_start = chrono::high_resolution_clock::now();

    //---------------Param and Key Generation---------------
    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(8192);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(8192));
    parms.set_plain_modulus(PLAINTEXT_MODULUS);
    auto context = SEALContext(parms);

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    auto secret_key = keygen.secret_key();
    // Parameter are large enough that we should be fine with these at max
    // decomposition
    GaloisKeys gal_keys;
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    keygen.create_galois_keys(gal_keys);

    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key); 
    BatchEncoder batch_encoder(context);
    size_t slot_count = batch_encoder.slot_count();

    time_end = chrono::high_resolution_clock::now();
    auto time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << " [" << time_diff.count() << " microseconds]\n";

    cout << "Preprocessing";
    time_start = chrono::high_resolution_clock::now();

    //---------------Preprocessing---------------
    int filter_h = (*filters)[0][0].rows();
    int filter_w = (*filters)[0][0].cols();
    auto [p_image, output_h, output_w] = pad_image(image, pad_valid,
                                                   filter_h, filter_w,
                                                   stride_h, stride_w);

    // Generate the column representation of image for convolution
    int channels = (*image).size();
    int output_channels = filters->size();
    const int col_height = filter_h * filter_w * channels;
    const int col_num = output_h * output_w;
    EChannel image_col(col_height, col_num);
    i2c(&p_image, &image_col, filter_h, filter_w, stride_h, stride_w, output_h, output_w);

    // Ciphertexts can be batched into two rows each of size slot_count / 2
    // So fit as many columns as possible into each half. 
    // This code won't work if col_height > slot_count / 2
    int pack_num = slot_count / (2 * col_height); 
    int num_cipher = ceil((float)col_num / (2*pack_num));

    //---------------Encoding/Encryption---------------
    // Can be plaintext since server is evaluating
    vector<Plaintext> filters_encoded(output_channels);
    for (int i = 0; i < output_channels; i++) {
        EChannel filter_col(1, col_height);
        // Use im2col with a filter size 1x1 to flatten
        i2c(&(*filters)[i], &filter_col, 1, 1, 1, 1, filter_h, filter_w);
        // Pack 2*pack_num copies of kernel into a vector for encoding
        vector<uint64_t> pod_matrix(slot_count, 0ULL);
        for (int j = 0; j < pack_num; j++) {
            copy(filter_col.data(), 
                 filter_col.data() + col_height,
                 pod_matrix.begin() + j * col_height);
            copy(filter_col.data(), 
                 filter_col.data() + col_height,
                 pod_matrix.begin() + (slot_count / 2) + j * col_height);;
        }
        batch_encoder.encode(pod_matrix, filters_encoded[i]);
    }

    // Encrypt each image column as a packed ciphertext
    vector<Ciphertext> image_enc(num_cipher);
    int j = 0;
    while (j < col_num) {
        vector<uint64_t> pod_matrix(slot_count, 0ULL);
        // Pack as many columns as possible
        int amt_to_pack = min(col_num - j, pack_num);
        for (int i = 0; i < amt_to_pack; i++, j++) {
            // Convert column to vector
            move(image_col.col(j).data(),
                 image_col.col(j).data() + col_height,
                 pod_matrix.begin() + (j % pack_num)*col_height);
        }
        // Do another loop for the second half of the vector
        amt_to_pack = min(col_num - j, pack_num);
        for (int i = 0; i < amt_to_pack; i++, j++) {
            move(image_col.col(j).data(),
                 image_col.col(j).data() + col_height,
                 pod_matrix.begin() + (slot_count/2) + (j%pack_num)*col_height);
        }
        // Encode and encrypt
        Plaintext batch_encoded;
        batch_encoder.encode(pod_matrix, batch_encoded);
        encryptor.encrypt(batch_encoded, image_enc[ceil((float)j / (2*pack_num))-1]);
        ciphertexts += 1;
    }

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << " [" << time_diff.count() << " microseconds]" << endl;
    
    cout << "\nTotal memory allocated from the current memory pool: "
            << (MemoryManager::GetPool().alloc_byte_count() >> 20) << " MB" << endl;

    auto total_time = chrono::high_resolution_clock::now();

    
    //---------------Online Evaluation---------------
    // For each kernel and column perform vector-matrix multiplication
    vector<Ciphertext> results_enc(num_cipher * output_channels);
    for (int k = 0; k < output_channels; k++) {
        cout << "Multiply and relinearize";
        time_start = chrono::high_resolution_clock::now();

        // Compute matrix multiplication for column and kernel
        vector<Ciphertext> mul_copies(num_cipher);
        for (int j = 0; j < num_cipher; j++) {
            evaluator.multiply_plain(image_enc[j], filters_encoded[k], mul_copies[j]);
            evaluator.relinearize_inplace(mul_copies[j], relin_keys);
            multiplications += 1;
        }

        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        cout << " [" << time_diff.count() << " microseconds]\n";

        cout << "Rotate and add";
        time_start = chrono::high_resolution_clock::now();

        // To acheive log(n) rotations + log(n) additions + log(n) subtractions
        //   -If ciphertext is odd, take the larger half and rotate it such that
        //    the extra element on the left half gets added twice. Store the
        //    rotated left half ciphertext.
        //   -If ciphertext is even, simply add the two halfs
        //   -Once you have the single rightmost term with the sum, subtract
        //   all the saved ciphertexts since this will remove the doubled terms
        int summed = 0;
        vector<Ciphertext> extra_terms;
        while (summed < col_height - 1) {
            int rotation = (col_height-summed)/2;
            Ciphertext rot_copy;
            for (int j = 0; j < num_cipher; j++) {
                evaluator.rotate_rows(mul_copies[j], -rotation, gal_keys, rot_copy);
                evaluator.add_inplace(mul_copies[j], rot_copy);
                if ((col_height - summed) % 2 != 0) {
                    extra_terms.push_back(rot_copy);
                }
                rotations += 1;
                additions += 1;
            }
            summed += rotation;
        }
        // extra_terms will have (num_cipher * vecs_to_subtract) elements
        for (int i = 0; i < extra_terms.size(); i++) {
            evaluator.sub_inplace(mul_copies[i % num_cipher], extra_terms[i]);
            subtractions += 1;
        }

        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        cout << " [" << time_diff.count() << " microseconds]\n";

        move(mul_copies.begin(),
             mul_copies.end(),
             results_enc.begin() + k*(num_cipher));
        }

    cout << "Decrypt";
    time_start = chrono::high_resolution_clock::now();
    
    //---------------Decryption---------------
    // Decrypt everything
    vector<uint64_t> results((2* pack_num * results_enc.size()));
    for (int i = 0; i < results_enc.size(); i++) {
        Plaintext plain;
        vector<uint64_t> pod_matrix(slot_count, 0ULL);
        decryptor.decrypt(results_enc[i], plain);
        batch_encoder.decode(plain, pod_matrix);
        // TODO: Clean this
        for (int j = 0; j < 2*pack_num; j++) {
            if (j >= pack_num) 
                results[i * 2 * pack_num + j] = pod_matrix[slot_count/2 + (j%pack_num+1)*col_height-1];
            else
                results[i * 2 * pack_num + j] = pod_matrix[(j+1)*col_height-1];
        }
    }

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << " [" << time_diff.count() << " microseconds]\n";
    
    // Reshape everything
    EImage final_result;
    for (int i = 0; i < output_channels; i++) {
        final_result.push_back(Map<Matrix<uint64_t, Dynamic, Dynamic, RowMajor>>(&results[i*(2*pack_num*num_cipher)], output_h, output_w));
    }
    
    cout << "\n\nCiphertexts: " << ciphertexts << endl;
    cout << "Additions: " << additions << endl;
    cout << "Subtractions: " << subtractions << endl;
    cout << "Rotations: " << rotations << endl;
    cout << "Multiplications: " << multiplications << endl;

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - total_time);
    cout << "Online Total [" << time_diff.count() << " microseconds]\n";

    return final_result;
}
template void print_image<EChannel>(EChannel *data);
template void print_image<EImage>(EImage *data);
template void print_image<EFilters>(EFilters *data);
