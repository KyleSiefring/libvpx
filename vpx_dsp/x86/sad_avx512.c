/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <immintrin.h>
#include "./vpx_dsp_rtcd.h"
#include "vpx_ports/mem.h"

#define FSAD64_H(h)                                                           \
  unsigned int vpx_sad64x##h##_avx512(const uint8_t *src_ptr, int src_stride, \
                                    const uint8_t *ref_ptr, int ref_stride) { \
    int i, res;                                                               \
    __m512i sad1_reg, sad2_reg, ref1_reg, ref2_reg;                           \
    __m512i sum_sad = _mm512_setzero_si512();                                 \
    __m512i sum_sad_h;                                                        \
    __m256i sum_sad256;                                                       \
    __m128i sum_sad128;                                                       \
    int ref2_stride = ref_stride << 1;                                        \
    int src2_stride = src_stride << 1;                                        \
    int max = h >> 1;                                                         \
    for (i = 0; i < max; i++) {                                               \
      ref1_reg = _mm512_loadu_si512((__m512i const *)ref_ptr);                \
      ref2_reg = _mm512_loadu_si512((__m512i const *)(ref_ptr + ref_stride)); \
      sad1_reg = _mm512_sad_epu8(                                             \
          ref1_reg, _mm512_loadu_si512((__m512i const *)src_ptr));            \
      sad2_reg = _mm512_sad_epu8(                                             \
          ref2_reg,                                                           \
          _mm512_loadu_si512((__m512i const *)(src_ptr + src_stride)));       \
      sum_sad =                                                               \
          _mm512_add_epi32(sum_sad, _mm512_add_epi32(sad1_reg, sad2_reg));    \
      ref_ptr += ref2_stride;                                                 \
      src_ptr += src2_stride;                                                 \
    }                                                                         \
    sum_sad_h = _mm512_bslli_epi128(sum_sad, 8);                              \
    sum_sad = _mm512_add_epi32(sum_sad, sum_sad_h);                           \
    sum_sad256 = _mm512_extracti32x8_epi32(sum_sad, 1);                       \
    sum_sad256 = _mm256_add_epi32(_mm512_castsi512_si256(sum_sad),            \
                                  sum_sad256);                                \
    sum_sad128 = _mm256_extracti128_si256(sum_sad256, 1);                     \
    sum_sad128 = _mm_add_epi32(_mm256_castsi256_si128(sum_sad256),            \
                               sum_sad128);                                   \
    res = _mm_cvtsi128_si32(sum_sad128);                                      \
    return res;                                                               \
  }

#define FSAD64  \
  FSAD64_H(64); \
  FSAD64_H(32);

FSAD64;

#undef FSAD64
#undef FSAD64_H

#define FSADAVG64_H(h)                                                        \
  unsigned int vpx_sad64x##h##_avg_avx512(                                    \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,         \
      int ref_stride, const uint8_t *second_pred) {                           \
    int i, res;                                                               \
    __m512i sad1_reg, sad2_reg, ref1_reg, ref2_reg;                           \
    __m512i sum_sad = _mm512_setzero_si512();                                 \
    __m512i sum_sad_h;                                                        \
    __m256i sum_sad256;                                                       \
    __m128i sum_sad128;                                                       \
    int ref2_stride = ref_stride << 1;                                        \
    int src2_stride = src_stride << 1;                                        \
    int max = h >> 1;                                                         \
    for (i = 0; i < max; i++) {                                               \
      ref1_reg = _mm512_loadu_si512((__m512i const *)ref_ptr);                \
      ref2_reg = _mm512_loadu_si512((__m512i const *)(ref_ptr + ref_stride)); \
      ref1_reg = _mm512_avg_epu8(                                             \
          ref1_reg, _mm512_loadu_si512((__m512i const *)second_pred));        \
      ref2_reg = _mm512_avg_epu8(                                             \
          ref2_reg, _mm512_loadu_si512((__m512i const *)(second_pred + 64))); \
      sad1_reg = _mm512_sad_epu8(                                             \
          ref1_reg, _mm512_loadu_si512((__m512i const *)src_ptr));            \
      sad2_reg = _mm512_sad_epu8(                                             \
          ref2_reg,                                                           \
          _mm512_loadu_si512((__m512i const *)(src_ptr + src_stride)));       \
      sum_sad =                                                               \
          _mm512_add_epi32(sum_sad, _mm512_add_epi32(sad1_reg, sad2_reg));    \
      ref_ptr += ref2_stride;                                                 \
      src_ptr += src2_stride;                                                 \
      second_pred += 128;                                                     \
    }                                                                         \
    sum_sad_h = _mm512_bslli_epi128(sum_sad, 8);                              \
    sum_sad = _mm512_add_epi32(sum_sad, sum_sad_h);                           \
    sum_sad256 = _mm512_extracti32x8_epi32(sum_sad, 1);                       \
    sum_sad256 = _mm256_add_epi32(_mm512_castsi512_si256(sum_sad),            \
                                  sum_sad256);                                \
    sum_sad128 = _mm256_extracti128_si256(sum_sad256, 1);                     \
    sum_sad128 = _mm_add_epi32(_mm256_castsi256_si128(sum_sad256),            \
                               sum_sad128);                                   \
    res = _mm_cvtsi128_si32(sum_sad128);                                      \
    return res;                                                               \
  }

#define FSADAVG64  \
  FSADAVG64_H(64); \
  FSADAVG64_H(32);

FSADAVG64;

#undef FSADAVG64
#undef FSADAVG64_H
