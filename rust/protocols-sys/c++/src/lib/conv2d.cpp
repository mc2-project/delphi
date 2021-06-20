#include <iostream>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <string>
#include <math.h>
#include <assert.h>
#include "seal/seal.h"
#include "conv2d.h"

#include <bitset>

using namespace seal;
using namespace std;

int ciphertexts = 0;
int result_ciphertexts = 0;
int rot_count = 0;
int multiplications = 0;
int additions = 0;
int subtractions = 0;

/* Helper function for rounding to the next power of 2
 * Credit: https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2 */
inline int next_pow2(int val) {
    return pow(2, ceil(log(val)/log(2)));
}

void print(Image &image, int chans, int image_h, int image_w) {
    cout << "------IMAGE------" << endl;
    for (int chan = 0; chan < chans; chan++) {
        for (int row = 0; row < image_h; row++) {
            cout << " [ ";
            int col = 0;
            for (; col < image_w-1; col++)
                cout << image[chan][row*image_h + col] << ", ";
            cout << image[chan][row*image_h + col] << " ],"<< endl;
        }
        cout << endl;
        cout << "-----------------" << endl;
    }
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
    for (int chan = 0; chan < 8; chan++) {
        for (int i = 0; i < 25; i++)
            cout << pod_matrix[chan*data.image_size + i] << ", ";
        cout << " || ";
    }
    cout << endl;
}


/* Helper function for plaintext rotations */
uv64 pt_rotate(int slot_count, int rotation, vector<u64> &vec) {
    uv64 new_vec(slot_count, 0);
    int pack_num = slot_count / 2;
    for (int half = 0; half < 2; half++) {
        for (int idx = 0; idx < pack_num; idx++) {
            // Wrap around the half if we accidently pull too far
            int offset = neg_mod(rotation+idx, pack_num);
            new_vec[half*pack_num + idx] = vec[half*pack_num + offset];
        } 
    }
    return new_vec;
}


/* Generates a masking vector of random noise that will be applied to parts of the ciphertext
 * that contain leakage from the convolution */
vector<Plaintext> HE_preprocess_noise(const uint64_t* const* secret_share, const Metadata &data,
        BatchEncoder &batch_encoder) {
    // Create uniform distribution
    random_device rd;
    mt19937 engine(rd());
    uniform_int_distribution<u64> dist(0, PLAINTEXT_MODULUS);
    auto gen = [&dist, &engine](){
        return dist(engine);
    };
    // Sample randomness into vector
    vector<uv64> noise(data.out_ct, uv64(data.slot_count, 0ULL));
    for (auto &vec: noise) {
        generate(begin(vec), end(vec), gen);
    }

    // Puncture the vector with secret share where an actual convolution result value lives
    for (int out_c = 0; out_c < data.out_chans; out_c++) {
        int ct_idx = out_c / (2*data.chans_per_half);
        int half_idx = (out_c % (2*data.chans_per_half)) / data.chans_per_half;
        int half_off = out_c % data.chans_per_half;
        for (int col = 0; col < data.output_h; col++) {
            for (int row = 0; row < data.output_w; row++) {
                int noise_idx = half_idx * data.pack_num
                                + half_off * data.image_size
                                + col * data.stride_w * data.image_w
                                + row * data.stride_h;
                int share_idx = col * data.output_w + row ;
                noise[ct_idx][noise_idx] = secret_share[out_c][share_idx];
            }
        }
    }
    
    // Encrypt all the noise vectors
    vector<Plaintext> enc_noise(data.out_ct);
    for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
        batch_encoder.encode(noise[ct_idx], enc_noise[ct_idx]);
    }
    return enc_noise; 
}


vector<uv64> preprocess_image(Metadata data, const u64* const* image) {
    vector<uv64> ct(data.inp_ct, uv64(data.slot_count, 0));
    int inp_c = 0;
    for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
        int inp_c_limit = (ct_idx+1) * 2 * data.chans_per_half;
        for (;inp_c < data.inp_chans && inp_c < inp_c_limit; inp_c++) {
            // Calculate which half of ciphertext the output channel 
            // falls in and the offest from that half, 
            int half_idx = (inp_c % (2*data.chans_per_half)) / data.chans_per_half;
            int half_off = inp_c % data.chans_per_half;
            for (int row = 0; row < data.image_h; row++) {
                for (int col = 0; col < data.image_w; col++) {
                    int idx = half_idx * data.pack_num
                            + half_off * data.image_size
                            + row * data.image_h
                            + col;
                    ct[ct_idx][idx] = image[inp_c][row*data.image_h + col];
                }
            }
        }
    }
    return ct;
}


/* Evaluates the filter rotations necessary to convole an input. Essentially, think
 * about placing the filter in the top left corner of the padded image and sliding the
 * image over the filter in such a way that we capture which elements of filter
 * multiply with which elements of the image. We account for the zero padding by
 * zero-puncturing the masks. This function can evaluate plaintexts and
 * ciphertexts.
*/
template <class T>
vector<T> filter_rotations(T &input, const Metadata &data, Evaluator *evaluator, GaloisKeys *gal_keys) {
    vector<T> rotations(input.size(), T(data.filter_size));
    int pad_h = data.pad_t + data.pad_b; 
    int pad_w = data.pad_l + data.pad_r; 

    // This tells us how many filters fit on a single row of the padded image
    int f_per_row = data.image_w + pad_w - data.filter_w + 1;

    // This offset calculates rotations needed to bring filter from top left
    // corner of image to the top left corner of padded image
    int offset = f_per_row * data.pad_t + data.pad_l;
    
    // For each element of the filter, rotate the padded image s.t. the top
    // left position always contains the first element of the image it touches
    for (int f_row = 0; f_row < data.filter_h; f_row++) {
        int row_offset = f_row * data.image_w - offset;
        for (int f_col = 0; f_col < data.filter_w; f_col++) {
            int rot_amt = row_offset + f_col;
            for (int ct_idx = 0; ct_idx < input.size(); ct_idx++) {
                int idx = f_row*data.filter_w+f_col;
                // The constexpr is necessary so the generic type to be used
                // in branch
                if constexpr (is_same<T, vector<uv64>>::value) {
                    rotations[ct_idx][idx] = pt_rotate(data.slot_count,
                                                       rot_amt,
                                                       input[ct_idx]);
                } else {
                    evaluator->rotate_rows(input[ct_idx],
                                          rot_amt,
                                          *gal_keys,
                                          rotations[ct_idx][idx]); 
                    rot_count += 1;
                }
            }
        }
    }
    return rotations;
}


/* Encrypts the given input image and all of its rotations */
vector<vector<Ciphertext>> HE_encrypt_rotations(vector<vector<uv64>> &rotations,
        const Metadata &data, Encryptor &encryptor, BatchEncoder &batch_encoder) {
    vector<vector<Ciphertext>> enc_rots(rotations.size(),
                                        vector<Ciphertext>(rotations[0].size()));
    for (int ct_idx = 0; ct_idx < rotations.size(); ct_idx++) {
        for (int f = 0; f < rotations[0].size(); f++) {
            Plaintext tmp;
            batch_encoder.encode(rotations[ct_idx][f], tmp);
            encryptor.encrypt(tmp, enc_rots[ct_idx][f]);
        } 
    }
    return enc_rots;
}


/* Encrypts the given input image */
vector<Ciphertext> HE_encrypt(vector<uv64> &pt, const Metadata &data,
        Encryptor &encryptor, BatchEncoder &batch_encoder) {
    vector<Ciphertext> ct(pt.size());
    for (int ct_idx = 0; ct_idx < pt.size(); ct_idx++) {
        Plaintext tmp;
        batch_encoder.encode(pt[ct_idx], tmp);
        encryptor.encrypt(tmp, ct[ct_idx]);
    }
    return ct;
}


/* Creates filter masks for an image input that has been output packed.
 *
 * The actual values of the masks are set in the exact same manner as input
 * packing. The added complexity is packing the output channels in a manner
 * which minimizes the total number of rotations needed to add everything back
 * together while also negating any need for multiplying by 0 masks or other
 * weird manipulates.
 *
 * In order to accomplish this, the following invariant must always hold true:
 *   i) Each mask only contains channels that will be in the same half of the
 *      output
 *      - ie. Output channels should never wrap-around within the same half
 *        UNLESS out_halves < 1 and the half repeats
 *      - This ensures that we don't need to multiply by any masks to add
 *        everything together since each half only contains channels that
 *        will be in the same half in the output
 *
 * We split all the output channels into various halves, generate all possible
 * permutations of those halves, and finally generate inward rotations of
 * those halves. 
 *
 * If out_halves > 1, we pack each half tightly except for the last one. If the
 * last half is at ct_idx == 0 and is less than a quarter ciphertext, we pad it
 * to the nearest power of two and repeat it as many times as possible. This
 * allows rotations to reduce to a log factor for this half. Note that if
 * ct_idx == 1 we can't do this optimization because the rotations would be
 * different for the first half of the ciphertext.
 *
 * Must also make sure that the filters rotate to the right (the opposite of
 * input packing), since then rotations won't wrap around on a repeating
 * half where inp_chans % chans_per_half != 0. This enables a further
 * optimization for the repeating half.
 */
vector<vector<vector<Plaintext>>> HE_preprocess_filters(const u64* const* const* filters,
        const Metadata &data, BatchEncoder &batch_encoder) {
    // Mask is convolutions x cts per convolution x mask size
    vector<vector<vector<uv64>>> masks(
            data.convs,
            vector<vector<uv64>>(
                data.inp_ct,
                vector<uv64>(data.filter_size, uv64(data.slot_count, 0))));
    // Since a half in a permutation may have a variable number of rotations we
    // use this index to track where we are at in the masks tensor
    int conv_idx = 0;
    // Build each half permutation as well as it's inward rotations
    for (int perm = 0; perm < data.half_perms; perm += 2) {
        // We populate two different half permutations at a time (if we have at
        // least 2). The second permutation is what you'd get by flipping the
        // columns of the first
        for (int half = 0; half < data.inp_halves; half++) {
            int ct_idx = half / 2;
            int half_idx = half % 2;
            int inp_base = half * data.chans_per_half;

            // The output channel the current ct starts from
            int out_base = (((perm/2) + ct_idx)*2*data.chans_per_half) % data.out_mod;
            // If we're on the last output half, the first and last halves aren't
            // in the same ciphertext, and the last half has repeats, then do 
            // repeated packing and skip the second half
            bool last_out = ((out_base + data.out_in_last) == data.out_chans)
                            && data.out_halves != 2;
            bool half_repeats = last_out && data.last_repeats;
            // If the half is repeating we do possibly less number of rotations
            int total_rots = (half_repeats) ? data.last_rots : data.half_rots;
            // Generate all inward rotations of each half
            for (int rot = 0; rot < total_rots; rot++) {
                for (int chan = 0; chan < data.chans_per_half
                                   && (chan + inp_base) < data.inp_chans; chan++) {
                    for (int f = 0; f < data.filter_size; f++) {
                        // Pull the value of this mask
                        int f_w = f % data.filter_w;
                        int f_h = f / data.filter_w;
                        // Set the coefficients of this channel for both
                        // permutations
                        u64 val, val2;
                        int out_idx, out_idx2;

                        // If this is a repeating half we first pad out_chans to
                        // nearest power of 2 before repeating
                        if (half_repeats) {
                            out_idx = neg_mod(chan-rot, data.repeat_chans) + out_base;
                            // If we're on a padding channel then val should be 0
                            val = (out_idx < data.out_chans)
                                ? filters[out_idx][inp_base + chan][f] : 0;
                            // Second value will always be 0 since the second
                            // half is empty if we are repeating
                            val2 = 0;
                        } else {
                            int offset = neg_mod(chan-rot, data.chans_per_half);
                            if (half_idx) {
                                // If out_halves < 1 we may repeat within a
                                // ciphertext
                                // TODO: Add the log optimization for this case
                                if (data.out_halves > 1)
                                    out_idx = offset + out_base + data.chans_per_half;
                                else
                                    out_idx = offset + out_base;
                                out_idx2 = offset + out_base;
                            } else {
                                out_idx = offset + out_base;
                                out_idx2 = offset + out_base + data.chans_per_half;
                            }
                            val = (out_idx < data.out_chans)
                                ? filters[out_idx][inp_base+chan][f] : 0;
                            val2 = (out_idx2 < data.out_chans)
                                ? filters[out_idx2][inp_base+chan][f] : 0;
                        }
                        // Iterate through the whole image and figure out which
                        // values the filter value touches - this is the same
                        // as for input packing
                        for(int curr_h = 0; curr_h < data.image_h; curr_h += data.stride_h) {
                            for(int curr_w = 0; curr_w < data.image_w; curr_w += data.stride_w) {
                                // curr_h and curr_w simulate the current top-left position of 
                                // the filter. This detects whether the filter would fit over
                                // this section. If it's out-of-bounds we set the mask index to 0
                                bool zero = ((curr_w+f_w) < data.pad_l) ||
                                    ((curr_w+f_w) >= (data.image_w+data.pad_l)) ||
                                    ((curr_h+f_h) < data.pad_t) ||
                                    ((curr_h+f_h) >= (data.image_h+data.pad_l));
                                // Calculate which half of ciphertext the output channel 
                                // falls in and the offest from that half, 
                                int idx = half_idx * data.pack_num
                                        + chan * data.image_size
                                        + curr_h * data.image_h
                                        + curr_w;
                                // Add both values to appropiate permutations
                                masks[conv_idx+rot][ct_idx][f][idx] = zero? 0: val;
                                if (data.half_perms > 1) {
                                    masks[conv_idx+data.half_rots+rot][ct_idx][f][idx] = zero? 0: val2;
                                }
                            }
                        }
                    }
                }
            }
        }
        conv_idx += 2*data.half_rots;
    }

    // Encode all the masks
    vector<vector<vector<Plaintext>>> encoded_masks(
            data.convs,
            vector<vector<Plaintext>>(
                data.inp_ct,
                vector<Plaintext>(data.filter_size)));
    for (int conv = 0; conv < data.convs; conv++) {
        for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
            for (int f = 0; f < data.filter_size; f++) {
                batch_encoder.encode(masks[conv][ct_idx][f],
                                         encoded_masks[conv][ct_idx][f]);
            } 
        } 
    }
    return encoded_masks;
}

vector<vector<Ciphertext>> HE_conv(vector<vector<vector<Plaintext>>> &masks,
        vector<vector<Ciphertext>> &rotations, const Metadata &data, Evaluator &evaluator,
        RelinKeys &relin_keys, Ciphertext &zero) {
    vector<vector<Ciphertext>> result(data.convs, vector<Ciphertext>(data.inp_ct));
    // Init the result vector to all 0
    for (int conv = 0; conv < data.convs; conv++) {
        for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) 
            result[conv][ct_idx] = zero;
    }
    // Multiply masks and add for each convolution
    for (int perm = 0; perm < data.half_perms; perm++) {
        for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
            // The output channel the current ct starts from
            int out_base = ((perm/2+ct_idx)*2*data.chans_per_half) % data.out_mod;
            // If we're on the last output half, the first and last halves aren't
            // in the same ciphertext, and the last half has repeats, then only 
            // convolve last_rots number of times
            bool last_out = ((out_base + data.out_in_last) == data.out_chans)
                            && data.out_halves != 2;
            bool half_repeats = last_out && data.last_repeats;
            int total_rots = (half_repeats) ? data.last_rots : data.half_rots;
            for (int rot = 0; rot < total_rots; rot++) {
                for (int f = 0; f < data.filter_size; f++) {
                    // Note that if a mask is zero this will result in a
                    // 'transparent' ciphertext which SEAL won't allow by default.
                    // This isn't a problem however since we're adding the result
                    // with something else, and the size is known beforehand so
                    // having some elements be 0 doesn't matter
                    Ciphertext tmp;
                    evaluator.multiply_plain(rotations[ct_idx][f],
                                                 masks[perm*data.half_rots+rot][ct_idx][f],
                                                 tmp);
                    evaluator.relinearize_inplace(tmp, relin_keys);
                    multiplications += 1;
                    evaluator.add_inplace(result[perm*data.half_rots+rot][ct_idx], tmp);
                    additions += 1;
                }
            }
        }
    }
    return result;
}

/* Takes the result of an output-packed convolution, and rotates + adds all the
 * ciphertexts to get a tightly packed output
 * 
 * The result of the convolution will be a vector of dimensions:
 *   (half_perms * half_rots) x inp_ct 
 * 
 * First, we sum up all the inward rotations of each half permutation. This is
 * done either by (chans_per_half-1) rotations + additions, or a log factor
 * (this factor is variable on how many inp/out channels are in the last half)
 * rotations + additions. This reduces the vector to dimensions:
 *   half_perms x inp_ct
 *
 * Finally, we rotate all the half permutations into their appropiate position
 * in the final output reducing the vector to dimensions:
 *   inp_ct
 * */
vector<Ciphertext> HE_output_rotations(vector<vector<Ciphertext>> convs,
        const Metadata &data, Evaluator &evaluator, GaloisKeys &gal_keys,
        Ciphertext &zero) {
    vector<vector<Ciphertext>> partials(data.half_perms,
                                        vector<Ciphertext>(data.inp_ct));
    // Init the result vector to all 0
    vector<Ciphertext> result(data.out_ct);
    for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
        result[ct_idx] = zero;
    }
    // For each half perm, add up all the inside channels of each half 
    for (int perm = 0; perm < data.half_perms; perm+=2) {
        int rot;
        // Can save an addition or so by initially setting the partials vector
        // to a convolution result if it's correctly aligned. Otherwise init to
        // all 0s
        if (data.inp_chans <= data.out_chans || data.out_chans == 1) {
            for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
                partials[perm][ct_idx] = convs[perm*data.half_rots][ct_idx];
                if (data.half_perms > 1) 
                    partials[perm+1][ct_idx] = convs[(perm+1)*data.half_rots][ct_idx];;
            }
            rot = 1;
        } else {
            for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
                partials[perm][ct_idx] = zero;
                if (data.half_perms > 1) 
                    partials[perm+1][ct_idx] = zero;
            }
            rot = 0;
        }
        for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
            // The output channel the current ct starts from
            int out_base = ((perm/2+ct_idx)*2*data.chans_per_half) % data.out_mod;
            // Whether we are on the last input half
            bool last_in = (perm + ct_idx + 1) % (data.inp_ct) == 0;
            // If we're on the last output half, the first and last halves aren't
            // in the same ciphertext, and the last half has repeats, then do the
            // rotations optimization when summing up 
            bool last_out = ((out_base + data.out_in_last) == data.out_chans)
                            && data.out_halves != 2;
            bool half_repeats = last_out && data.last_repeats;
            int total_rots = (half_repeats) ? data.last_rots : data.half_rots;
            for (int in_rot = rot; in_rot < total_rots; in_rot++) {
                int conv_idx = perm * data.half_rots + in_rot;
                int rot_amt;
                // If we're on a repeating half the amount we rotate will be
                // different
                if (half_repeats)
                    rot_amt = -neg_mod(-in_rot, data.repeat_chans) * data.image_size;
                else 
                    rot_amt = -neg_mod(-in_rot, data.chans_per_half) * data.image_size;
                
                evaluator.rotate_rows_inplace(convs[conv_idx][ct_idx],
                                                  rot_amt,
                                                  gal_keys);
                evaluator.add_inplace(partials[perm][ct_idx],
                                          convs[conv_idx][ct_idx]);
                // Do the same for the column swap if it exists
                if (data.half_perms > 1) {
                    evaluator.rotate_rows_inplace(convs[conv_idx+data.half_rots][ct_idx],
                                                      rot_amt,
                                                      gal_keys);
                    evaluator.add_inplace(partials[perm+1][ct_idx],
                                              convs[conv_idx+data.half_rots][ct_idx]);
                    if (rot_amt != 0)
                        rot_count += 1;
                    additions += 1;
                }
                if (rot_amt != 0)
                    rot_count += 1;
                additions += 1;
            }
            // Add up a repeating half
            if (half_repeats) {
                // If we're on the last inp_half then we might be able to do
                // less rotations. We may be able to find a power of 2 less
                // than chans_per_half that contains all of our needed repeats
                int size_to_reduce;
                if (last_in) {
                    int num_repeats = ceil((float) data.inp_in_last / data.repeat_chans);
                    //  We round the repeats to the closest power of 2
                    int effective_repeats;
                    // When we rotated in the previous loop we cause a bit of overflow 
                    // (one extra repeat_chans worth). If the extra overflow fits 
                    // into the modulo of the last repeat_chan we can do one
                    // less rotation
                    if (data.repeat_chans*num_repeats % data.inp_in_last == data.repeat_chans - 1)
                        effective_repeats = pow(2, ceil(log(num_repeats)/log(2)));
                    else
                        effective_repeats = pow(2, ceil(log(num_repeats+1)/log(2)));
                    // If the overflow ended up wrapping around then we simply
                    // want chans_per_half as our size
                    size_to_reduce = min(effective_repeats*data.repeat_chans, data.chans_per_half);
                } else 
                    size_to_reduce = data.chans_per_half;
                // Perform the actual rotations
                for (int in_rot = size_to_reduce/2; in_rot >= data.repeat_chans; in_rot = in_rot/2) {
                    int rot_amt = in_rot * data.image_size;
                    Ciphertext tmp = partials[perm][ct_idx];
                    evaluator.rotate_rows_inplace(tmp, rot_amt, gal_keys);
                    evaluator.add_inplace(partials[perm][ct_idx], tmp);
                    // Do the same for column swap if exists
                    if (data.half_perms > 1) {
                        tmp = partials[perm+1][ct_idx];
                        evaluator.rotate_rows_inplace(tmp, rot_amt, gal_keys);
                        evaluator.add_inplace(partials[perm+1][ct_idx], tmp);
                        if (rot_amt != 0)
                            rot_count += 1;
                        additions += 1;
                    }
                    if (rot_amt != 0)
                        rot_count += 1;
                    additions += 1;
                }
            }
            // The correct index for the correct ciphertext in the final output
            int out_idx = (perm/2 + ct_idx) % data.out_ct;
            if (perm == 0) {
                // The first set of convolutions is aligned correctly
                evaluator.add_inplace(result[out_idx], partials[perm][ct_idx]);
                if (data.out_halves == 1 && data.inp_halves > 1) {
                    // If the output fits in a single half but the input
                    // doesn't, add the two columns
                    evaluator.rotate_columns_inplace(partials[perm][ct_idx],
                                                         gal_keys);
                    evaluator.add_inplace(result[out_idx], partials[perm][ct_idx]);
                    rot_count += 1;
                    additions += 1;
                } 
                // Do the same for column swap if exists and we aren't on a repeat
                if (data.half_perms > 1 && !half_repeats) {
                    evaluator.rotate_columns_inplace(partials[perm+1][ct_idx],
                                                         gal_keys);
                    evaluator.add_inplace(result[out_idx], partials[perm+1][ct_idx]);
                    rot_count += 1;
                    additions += 1;
                }
            } else {
                // Rotate the output ciphertexts by one and add
                evaluator.add_inplace(result[out_idx], partials[perm][ct_idx]);
                additions += 1;
                // If we're on a tight half we add both halves together and
                // don't look at the column flip
                if (half_repeats) {
                    evaluator.rotate_columns_inplace(partials[perm][ct_idx],
                                                         gal_keys);
                    evaluator.add_inplace(result[out_idx], partials[perm][ct_idx]);
                    rot_count += 1;
                    additions += 1;
                } else if (data.half_perms > 1) {
                    evaluator.rotate_columns_inplace(partials[perm+1][ct_idx],
                                                         gal_keys);
                    evaluator.add_inplace(result[out_idx], partials[perm+1][ct_idx]);
                    rot_count += 1;
                    additions += 1;
                }
            }
        }
    }
    return result;
}


/* Decrypts and reshapes convolution result */
u64** HE_decrypt(vector<Ciphertext> &enc_result, const Metadata &data, Decryptor &decryptor,
        BatchEncoder &batch_encoder) {
    // Decrypt ciphertext 
    vector<vector<u64>> result(data.out_ct);
    for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
        Plaintext tmp;
        decryptor.decrypt(enc_result[ct_idx], tmp);
        batch_encoder.decode(tmp, result[ct_idx]);
    }
    
    u64** final_result = new u64*[data.out_chans];
    // Extract correct values to reshape
    for (int out_c = 0; out_c < data.out_chans; out_c++) {
        int ct_idx = out_c / (2*data.chans_per_half);
        int half_idx = (out_c % (2*data.chans_per_half)) / data.chans_per_half;
        int half_off = out_c % data.chans_per_half;
        // Depending on the padding type and stride the output values won't be
        // lined up so extract them into a temporary channel before placing
        // them in resultant Image
        final_result[out_c] = new u64[data.output_h*data.output_w];
        for (int col = 0; col < data.output_h; col++) {
            for (int row = 0; row < data.output_w; row++) {
                int idx = half_idx * data.pack_num
                        + half_off * data.image_size
                        + col * data.stride_w * data.image_w
                        + row * data.stride_h;
                final_result[out_c][col*data.output_w + row] = result[ct_idx][idx];
            }
        }
    }
    return final_result; 
}


/* Populates the Metadata struct */
Metadata conv_metadata(int slot_count, int image_h, int image_w, int filter_h, int filter_w,
        int inp_chans, int out_chans, int stride_h, int stride_w, bool pad_valid) {
    // If using Output packing we pad image_size to the nearest power of 2
    int image_size = next_pow2(image_h*image_w);
    int filter_size = filter_h * filter_w;

    int pack_num = slot_count / 2;
    int chans_per_half = pack_num / image_size;
    int out_ct = ceil((float) out_chans / (2*chans_per_half));
    // In input packing we create inp_chans number of ciphertexts that are the
    // size of the output. In output packing we simply send the input
    // ciphertext
    int inp_ct = ceil((float) inp_chans / (2*chans_per_half));


    int inp_halves = ceil((float) inp_chans / chans_per_half);
    int out_halves = ceil((float) out_chans / chans_per_half);

    // The modulo is calculated per ciphertext instead of per half since we
    // should never have the last out_half wrap around to the first in the same
    // ciphertext (unless there's only one output ciphertext)
    int out_mod = out_ct*2*chans_per_half;
    // This should always be even unless the whole ciphertext fits in a single
    // half since for each ciphertext permutation you also have to flip the
    // columns to get full coverage.
    int half_perms = (out_halves % 2 != 0 && out_halves > 1) ? out_halves + 1 : out_halves;

    assert(out_chans > 0 && inp_chans > 0);
    // Doesn't currently support a channel being larger than a half ciphertext
    assert(image_size < (slot_count/2));

    // Number of inp/out channels in the last half
    int out_in_last = (out_chans % chans_per_half) ?
        (out_chans % (2*chans_per_half)) % chans_per_half : chans_per_half;
    int inp_in_last = (inp_chans % chans_per_half) ?
        (inp_chans % (2*chans_per_half)) % chans_per_half : chans_per_half;

    // Pad repeating channels in last half to the nearest power of 2 to allow
    // for a log number of rotations
    int repeat_chans = next_pow2(out_in_last);
    // If the out_chans in the last half take up a quarter ciphertext or less,
    // and the last half is the only half in the last ciphertext, we can do
    // repeats with log # of rotations
    bool last_repeats = (out_in_last <= chans_per_half/2) && (out_halves % 2 == 1);
   
    // If the input or output is greater than one half ciphertext it will be tightly
    // packed, otherwise the number of rotations depends on the max of output
    // or inp chans 
    int half_rots = (inp_halves > 1 || out_halves > 1) ? 
        chans_per_half : max(max(out_chans, inp_chans), chans_per_half);
    // If we have repeats in the last half, we may have less rotations
    int last_rots = (last_repeats) ? repeat_chans : half_rots;

    // If we have only have a single half than we can possibly do less
    // convolutions if that half has repeats. Note that if a half has repeats
    // (ie. less rotations for a convolution) but we still do the half_rots #
    // of rotations than we simply skip the convolution for the last half
    // appropiately
    int convs = (out_halves == 1) ? last_rots : half_perms * half_rots;

    // Calculate padding
    int output_h, output_w, pad_t, pad_b, pad_r, pad_l;

    if (pad_valid) {
        output_h = ceil((float)(image_h - filter_h + 1) / stride_h);
        output_w = ceil((float)(image_w - filter_w + 1) / stride_w);
        pad_t = 0;
        pad_b = 0;
        pad_r = 0;
        pad_l = 0;
    } else {
        output_h = ceil((float)image_h / stride_h);
        output_w = ceil((float)image_w / stride_w);
        // Total amount of vertical and horizontal padding needed
        int pad_h = max((output_h - 1) * stride_h + filter_h - image_h, 0);
        int pad_w = max((output_w - 1) * stride_w + filter_w - image_w, 0);
        // Individual side padding
        pad_t = floor((float)pad_h / 2);
        pad_b = pad_h - pad_t;
        pad_l = floor((float)pad_w / 2);
        pad_r = pad_w - pad_l;
    }
    Metadata data = {slot_count, pack_num, chans_per_half, inp_ct, out_ct,
        image_h, image_w, image_size, inp_chans, filter_h, filter_w,
        filter_size, out_chans, inp_halves, out_halves, out_in_last, inp_in_last,
        out_mod, half_perms, last_repeats, repeat_chans, half_rots, last_rots,
        convs, stride_h, stride_w, output_h, output_w, pad_t, pad_b, pad_r, pad_l};
    return data;
}

// These are so linking doesn't fail
template vector<vector<Ciphertext>> filter_rotations<vector<Ciphertext>>(vector<Ciphertext> &input, const Metadata &data,
    Evaluator *evaluator, GaloisKeys *gal_keys);
template vector<vector<uv64>> filter_rotations<vector<uv64>>(vector<uv64> &input, const Metadata &data,
    Evaluator *evaluator, GaloisKeys *gal_keys);
