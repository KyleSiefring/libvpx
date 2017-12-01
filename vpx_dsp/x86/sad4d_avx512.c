/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <immintrin.h>  // AVX512

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"

#include "iacaMarks.h"

//Nope :/
void vpx_sad4x4x4d_avx512(const uint8_t *src, int src_stride,
                          const uint8_t *const ref[4], int ref_stride,
                          uint32_t res[4]) {
//IACA_START
#if ARCH_X86_64
  __m256i src_reg, ref_reg;
  __m256i sum_reg;
  __m512i gather_reg;
  int i;
  uint8_t *offset = 0;
  {
    __m256i lo, hi;
    // 0 1 2 3
    __m256i ptrs = _mm256_loadu_si256((const __m256i *)ref);
    __m256i stride = _mm256_broadcastq_epi64(_mm_cvtsi32_si128(ref_stride));
    stride = _mm256_add_epi64(stride, ptrs);
    // 0 0 2 2
    lo = _mm256_unpacklo_epi64(ptrs, stride);
    // 1 1 3 3
    hi = _mm256_unpacklo_epi64(ptrs, stride);
    gather_reg = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
  }
  sum_reg = _mm256_set1_epi16(0);
  for (i = 0; i < 4; i += 2) {
    __m128i src_reg_128 = _mm_cvtsi32_si128(*(int const *)(src));
    src_reg_128 = _mm_insert_epi32(src_reg_128, *(int const *)(src + src_stride), 1);
    src_reg = _mm256_broadcast_i32x2(src_reg_128);
    ref_reg = _mm512_i64gather_epi32(gather_reg, (void *)offset, 1);
    ref_reg = _mm256_sad_epu8(ref_reg, src_reg);
    sum_reg = _mm256_add_epi32(sum_reg, ref_reg);
    src += 2*src_stride;
    offset += 2*ref_stride;
  }
  {
    __m128i sum128;
    const __m256i idx = _mm256_castsi128_si256(_mm_setr_epi32(0, 4, 2, 6));
    sum128 = _mm256_castsi256_si128(_mm256_permutex2var_epi32(sum_reg, idx, sum_reg));
    //sum128 = _mm256_cvtepi64_epi32(sum_reg);
    _mm_storeu_si128((__m128i *)(res), sum128);
  }
#else
#endif
//IACA_END
}

//Nope :(
void vpx_sad8x8x4d_avx512(const uint8_t *src, int src_stride,
                          const uint8_t *const ref[4], int ref_stride,
                          uint32_t res[4]) {
//IACA_START
#if ARCH_X86_64
  __m512i src_reg, ref_reg;
  __m512i sum_reg;
  __m512i gather_reg;
  int i;
  uint8_t *offset = 0;
  {
    const __m256i zero_reg = _mm256_set1_epi16(0);
    __m256i ptrs = _mm256_loadu_si256((const __m256i *)ref);
    __m256i stride = _mm256_broadcastq_epi64(_mm_cvtsi32_si128(ref_stride));
    // try 0 1  0 1  1 0  1 0 to remove broadcast
#if 1
    stride = _mm256_add_epi64(stride, ptrs);
    gather_reg = _mm512_inserti64x4(_mm512_castsi256_si512(ptrs), stride, 1);
#else
    __m256i lo = _mm256_unpacklo_epi64(zero_reg, stride);
    __m256i hi = _mm256_unpacklo_epi64(stride, zero_reg);
    gather_reg = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
#endif
  }
  sum_reg = _mm512_set1_epi16(0);
  for (i = 0; i < 8; i += 2) {
#if 1
    src_reg = _mm512_inserti64x4(_mm512_castsi256_si512(
      _mm256_broadcastq_epi64(_mm_loadl_epi64((const __m128i *)src))),
      _mm256_broadcastq_epi64(_mm_loadl_epi64((const __m128i *)(src + src_stride))), 1);
#else
    __m128i src_reg_128 = _mm_loadl_epi64((const __m128i *)src);
    src_reg_128 = _mm_insert_epi64(src_reg_128, ((long long const *)(src + src_stride))[0], 1);
    src_reg = _mm512_broadcast_i64x2(src_reg_128);
#endif
    ref_reg = _mm512_i64gather_epi64(gather_reg, (void *)offset, 1);
    ref_reg = _mm512_sad_epu8(ref_reg, src_reg);
    sum_reg = _mm512_add_epi32(sum_reg, ref_reg);
    src += 2*src_stride;
    offset += 2*ref_stride;
  }
  {
    __m256i sum256;
    __m128i sum128;
    sum256 = _mm512_cvtepi64_epi32(sum_reg);
    sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum256),
                           _mm256_extractf128_si256(sum256, 1));
    _mm_storeu_si128((__m128i *)(res), sum128);
  }
#else
#endif
//IACA_END
}

void vpx_sad64x64x4d_avx512(const uint8_t *src, int src_stride,
                            const uint8_t *const ref[4], int ref_stride,
                            uint32_t res[4]) {
  __m512i src_reg, ref0_reg, ref1_reg, ref2_reg, ref3_reg;
  __m512i sum_ref0, sum_ref1, sum_ref2, sum_ref3;
  __m512i sum_mlow, sum_mhigh;
  int i;
  const uint8_t *ref0, *ref1, *ref2, *ref3;

  ref0 = ref[0];
  ref1 = ref[1];
  ref2 = ref[2];
  ref3 = ref[3];
  sum_ref0 = _mm512_set1_epi16(0);
  sum_ref1 = _mm512_set1_epi16(0);
  sum_ref2 = _mm512_set1_epi16(0);
  sum_ref3 = _mm512_set1_epi16(0);
  for (i = 0; i < 64; i++) {
    // load src and all refs
    src_reg = _mm512_loadu_si512((const __m512i *)src);
    ref0_reg = _mm512_loadu_si512((const __m512i *)ref0);
    ref1_reg = _mm512_loadu_si512((const __m512i *)ref1);
    ref2_reg = _mm512_loadu_si512((const __m512i *)ref2);
    ref3_reg = _mm512_loadu_si512((const __m512i *)ref3);
    // sum of the absolute differences between every ref-i to src
    ref0_reg = _mm512_sad_epu8(ref0_reg, src_reg);
    ref1_reg = _mm512_sad_epu8(ref1_reg, src_reg);
    ref2_reg = _mm512_sad_epu8(ref2_reg, src_reg);
    ref3_reg = _mm512_sad_epu8(ref3_reg, src_reg);
    // sum every ref-i
    sum_ref0 = _mm512_add_epi32(sum_ref0, ref0_reg);
    sum_ref1 = _mm512_add_epi32(sum_ref1, ref1_reg);
    sum_ref2 = _mm512_add_epi32(sum_ref2, ref2_reg);
    sum_ref3 = _mm512_add_epi32(sum_ref3, ref3_reg);

    src += src_stride;
    ref0 += ref_stride;
    ref1 += ref_stride;
    ref2 += ref_stride;
    ref3 += ref_stride;
  }
  {
    __m256i sum256;
    __m128i sum128;
    // in sum_ref-i the result is saved in the first 4 bytes
    // the other 4 bytes are zeroed.
    // sum_ref1 and sum_ref3 are shifted left by 4 bytes
    sum_ref1 = _mm512_bslli_epi128(sum_ref1, 4);
    sum_ref3 = _mm512_bslli_epi128(sum_ref3, 4);

    // merge sum_ref0 and sum_ref1 also sum_ref2 and sum_ref3
    sum_ref0 = _mm512_or_si512(sum_ref0, sum_ref1);
    sum_ref2 = _mm512_or_si512(sum_ref2, sum_ref3);

    // merge every 64 bit from each sum_ref-i
    sum_mlow = _mm512_unpacklo_epi64(sum_ref0, sum_ref2);
    sum_mhigh = _mm512_unpackhi_epi64(sum_ref0, sum_ref2);

    // add the low 64 bit to the high 64 bit
    sum_mlow = _mm512_add_epi32(sum_mlow, sum_mhigh);

    // add the low 128 bit to the high 128 bit
    sum256 = _mm256_add_epi32(_mm512_castsi512_si256(sum_mlow),
                              _mm512_extracti32x8_epi32(sum_mlow, 1));
    sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum256),
                           _mm256_extractf128_si256(sum256, 1));

    _mm_storeu_si128((__m128i *)(res), sum128);
  }
}
