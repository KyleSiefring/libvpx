/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Usee of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <immintrin.h>  // AVX2

#include "./vp9_rtcd.h"
#include "vpx/vpx_integer.h"

int64_t vp9_block_error_avx2(const int16_t *coeff,
                             const int16_t *dqcoeff,
                             intptr_t block_size,
                             int64_t *ssz) {
  __m256i sse_reg, ssz_reg, coeff_reg1, dqcoeff_reg1, coeff_reg2, dqcoeff_reg2;
  __m256i exp_dqcoeff_lo, exp_dqcoeff_hi, exp_coeff_lo, exp_coeff_hi;
  __m256i sse_reg_64hi, ssz_reg_64hi;
  __m128i sse_reg128, ssz_reg128;
  int64_t sse;
  int i;
  const __m256i zero_reg = _mm256_set1_epi16(0);

  // if the block size is only 16 then use the code that loads only 32 bytes
  if (block_size == 16) {
    // load 32 bytes from coeff and dqcoeff
    coeff_reg1 = _mm256_loadu_si256((const __m256i *)(coeff));
    dqcoeff_reg1 = _mm256_loadu_si256((const __m256i *)(dqcoeff));
    // dqcoeff - coeff
    dqcoeff_reg1 = _mm256_sub_epi16(dqcoeff_reg1, coeff_reg1);
    // madd (dqcoeff - coeff)
    dqcoeff_reg1 = _mm256_madd_epi16(dqcoeff_reg1, dqcoeff_reg1);
    // madd coeff
    coeff_reg1 = _mm256_madd_epi16(coeff_reg1, coeff_reg1);
    // save the higher 64 bit of each 128 bit lane
    dqcoeff_reg2 = _mm256_srli_si256(dqcoeff_reg1, 8);
    coeff_reg2 = _mm256_srli_si256(coeff_reg1, 8);
    // add the higher 64 bit to the low 64 bit
    dqcoeff_reg1 = _mm256_add_epi32(dqcoeff_reg1, dqcoeff_reg2);
    coeff_reg1 = _mm256_add_epi32(coeff_reg1, coeff_reg2);
    // expand each double word in the lower 64 bits to quad word
    sse_reg = _mm256_unpacklo_epi32(dqcoeff_reg1, zero_reg);
    ssz_reg  = _mm256_unpacklo_epi32(coeff_reg1, zero_reg);
  }
  else {
    // init sse and ssz registerd to zero
    sse_reg = _mm256_set1_epi16(0);
    ssz_reg = _mm256_set1_epi16(0);

    for (i = 0; i < block_size; i+= 32) {
      // load 64 bytes from coeff and dqcoeff
      coeff_reg1 = _mm256_loadu_si256((const __m256i *)(coeff + i));
      dqcoeff_reg1 = _mm256_loadu_si256((const __m256i *)(dqcoeff + i));
      coeff_reg2 = _mm256_loadu_si256((const __m256i *)(coeff + i + 16));
      dqcoeff_reg2 = _mm256_loadu_si256((const __m256i *)(dqcoeff + i + 16));
      // dqcoeff - coeff
      dqcoeff_reg1 = _mm256_sub_epi16(dqcoeff_reg1, coeff_reg1);
      dqcoeff_reg2 = _mm256_sub_epi16(dqcoeff_reg2, coeff_reg2);
      // madd (dqcoeff - coeff)
      dqcoeff_reg1 = _mm256_madd_epi16(dqcoeff_reg1, dqcoeff_reg1);
      dqcoeff_reg2 = _mm256_madd_epi16(dqcoeff_reg2, dqcoeff_reg2);
      // madd coeff
      coeff_reg1 = _mm256_madd_epi16(coeff_reg1, coeff_reg1);
      coeff_reg2 = _mm256_madd_epi16(coeff_reg2, coeff_reg2);
      // add the first madd (dqcoeff - coeff) with the second
      dqcoeff_reg1 = _mm256_add_epi32(dqcoeff_reg1, dqcoeff_reg2);
      // add the first madd (coeff) with the second
      coeff_reg1 = _mm256_add_epi32(coeff_reg1, coeff_reg2);
      // expand each double word of madd (dqcoeff - coeff) to quad word
      exp_dqcoeff_lo = _mm256_unpacklo_epi32(dqcoeff_reg1, zero_reg);
      exp_dqcoeff_hi = _mm256_unpackhi_epi32(dqcoeff_reg1, zero_reg);
      // expand each double word of madd (coeff) to quad word
      exp_coeff_lo = _mm256_unpacklo_epi32(coeff_reg1, zero_reg);
      exp_coeff_hi = _mm256_unpackhi_epi32(coeff_reg1, zero_reg);
      // add each quad word of madd (dqcoeff - coeff) and madd (coeff)
      sse_reg = _mm256_add_epi64(sse_reg, exp_dqcoeff_lo);
      ssz_reg = _mm256_add_epi64(ssz_reg, exp_coeff_lo);
      sse_reg = _mm256_add_epi64(sse_reg, exp_dqcoeff_hi);
      ssz_reg = _mm256_add_epi64(ssz_reg, exp_coeff_hi);
    }
  }
  // save the higher 64 bit of each 128 bit lane
  sse_reg_64hi = _mm256_srli_si256(sse_reg, 8);
  ssz_reg_64hi = _mm256_srli_si256(ssz_reg, 8);
  // add the higher 64 bit to the low 64 bit
  sse_reg = _mm256_add_epi64(sse_reg, sse_reg_64hi);
  ssz_reg = _mm256_add_epi64(ssz_reg, ssz_reg_64hi);

  // add each 64 bit from each of the 128 bit lane of the 256 bit
  sse_reg128 = _mm_add_epi64(_mm256_castsi256_si128(sse_reg),
                             _mm256_extractf128_si256(sse_reg, 1));

  ssz_reg128 = _mm_add_epi64(_mm256_castsi256_si128(ssz_reg),
                             _mm256_extractf128_si256(ssz_reg, 1));

  // store the results
  _mm_storel_epi64((__m128i*)(&sse), sse_reg128);

  _mm_storel_epi64((__m128i*)(ssz), ssz_reg128);
  return sse;
}
