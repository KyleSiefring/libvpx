/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <immintrin.h>  // AVX2
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"

#include "vpx_dsp/x86/static.h"
#include "iacaMarks.h"

void vpx_sad16x16_xyz_1_avx2(const uint8_t *src, int src_stride,
                          const uint8_t *const ref, int ref_stride,
                          uint32_t res[8]) {
#define XYZ
  int i;
// TODO: Add set to zero
  __m256i src_reg_256;
#ifdef XYZ
  __m256i ref_reg_lo_256, ref_reg_hi_256;
  __m128i ref_reg_lo, ref_reg_hi;
  __m256i sum_ref_4_2, sum_ref_5_3, sum_ref_1_0, sum_ref_2_4, sum_ref_3_5;
  __m128i sum_ref_6, sum_ref_0, sum_ref_1, sum_ref_7;
#else
  __m128i ref_regs_lo[3];
  __m128i ref_regs_hi[3];
  __m128i sum_ref[8];
#endif
  const uint8_t *src_ptr = src;
  const uint8_t *ref_ptr = ref - ref_stride - 1;

#ifdef XYZ
  {
    __m256i a, b;
    a = _mm256_loadu_si256((const __m256i *)(ref_ptr + 0 * ref_stride));
    b = _mm256_loadu_si256((const __m256i *)(ref_ptr + 1 * ref_stride));
    ref_reg_lo_256 = _mm256_permute2x128_si256(a, b, 0x2);
    ref_reg_hi_256 = _mm256_permute2x128_si256(a, b, 0x7);
  }
#else
  ref_regs_lo[0] = _mm_loadu_si128((const __m128i *)(ref_ptr + 0 * ref_stride));
  ref_regs_lo[1] = _mm_loadu_si128((const __m128i *)(ref_ptr + 1 * ref_stride));
  ref_regs_hi[0] = _mm_loadu_si128((const __m128i *)(ref_ptr + 0 * ref_stride + 16));
  ref_regs_hi[1] = _mm_loadu_si128((const __m128i *)(ref_ptr + 1 * ref_stride + 16));
#endif
  for (i = 0; i < 16; i++) {
IACA_START
    __m128i tmp;
    src_reg_256 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)src_ptr));
#define src_reg (_mm256_castsi256_si128(src_reg_256))
#ifdef XYZ
#if 0
    ref_reg_lo = _mm_loadu_si128((const __m128i *)(ref_ptr + 2 * ref_stride));
    ref_reg_hi = _mm_loadu_si128((const __m128i *)(ref_ptr + 2 * ref_stride + 16));
    sum_ref_4_2 = _mm256_add_epi32(sum_ref_4_2, _mm256_sad_epu8(ref_reg_lo_256, src_reg_256));
    sum_ref_6 = _mm_add_epi32(sum_ref_6, _mm_sad_epu8(ref_reg_lo, src_reg));
#define ALIGNRX2_1(a, b) _mm256_alignr_epi8(a, b, 15)//alignr a, b, 1
#define ALIGNR_1(a, b) _mm_alignr_epi8(a, b, 15)//alignr a, b, 1
/*
    tmp = ALIGNR_1(_mm256_castsi256_si128(ref_reg_lo_256), _mm256_castsi256_si128(ref_regs_hi_256));
    sum_ref_0 = _mm_add_epi32(sum_ref_0, _mm_sad_epu8(tmp, src_reg));
    tmp = ALIGNR_1(ref_reg_lo, ref_reg_hi);
    sum_ref_1 = _mm_add_epi32(sum_ref_1, _mm_sad_epu8(tmp, src_reg));
*/
    {
      __m256i a, b;
      a = _mm256_blend_epi32(ref_reg_lo_256, _mm256_castsi128_si256(ref_reg_lo), 0xF);
      b = _mm256_blend_epi32(ref_reg_hi_256, _mm256_castsi128_si256(ref_reg_hi), 0xF);
      /*a = _mm256_inserti128_si256(ref_reg_lo_256, ref_reg_lo, 0);
      b = _mm256_inserti128_si256(ref_reg_hi_256, ref_reg_hi, 0);*/
      a = ALIGNRX2_1(a, b);
      sum_ref_1_0 = _mm256_add_epi32(sum_ref_1_0, _mm256_sad_epu8(a, src_reg_256));
    }
#define ALIGNRX2_2(a, b) _mm256_alignr_epi8(a, b, 14)
#define ALIGNR_2(a, b) _mm_alignr_epi8(a, b, 14)
    sum_ref_5_3 = _mm256_add_epi32(sum_ref_5_3, _mm256_sad_epu8(ALIGNRX2_2(ref_reg_lo_256, ref_reg_hi_256), src_reg_256));
    tmp = ALIGNR_2(ref_reg_lo, ref_reg_hi);
    sum_ref_7 = _mm_add_epi32(sum_ref_7, _mm_sad_epu8(tmp, src_reg));
    ref_reg_lo_256 = _mm256_permute2x128_si256(ref_reg_lo_256, _mm256_castsi128_si256(ref_reg_lo), 0x2);
    ref_reg_hi_256 = _mm256_permute2x128_si256(ref_reg_hi_256, _mm256_castsi128_si256(ref_reg_hi), 0x2);
#else


// sum_refs are labeled wrong- STILL WRONG
// 2 4 -> 6 4
// 3 5 -> 7 5
// 6 -> 2   7 -> 3
    ref_reg_lo = _mm_loadu_si128((const __m128i *)(ref_ptr + 2 * ref_stride));
    ref_reg_hi = _mm_loadu_si128((const __m128i *)(ref_ptr + 2 * ref_stride + 16));
    {
      __m256i a, b;
      a = _mm256_blend_epi32(ref_reg_lo_256, _mm256_castsi128_si256(ref_reg_lo), 0xF);
      b = _mm256_blend_epi32(ref_reg_hi_256, _mm256_castsi128_si256(ref_reg_hi), 0xF);
      ref_reg_lo = _mm256_castsi256_si128(ref_reg_lo_256);
      ref_reg_hi = _mm256_castsi256_si128(ref_reg_hi_256);
      ref_reg_lo_256 = a;
      ref_reg_lo_256 = b;
    }
    sum_ref_2_4 = _mm256_add_epi32(sum_ref_2_4, _mm256_sad_epu8(ref_reg_lo_256, src_reg_256));
    sum_ref_6 = _mm_add_epi32(sum_ref_6, _mm_sad_epu8(ref_reg_lo, src_reg));
#define ALIGNRX2_1(a, b) _mm256_alignr_epi8(a, b, 15)//alignr a, b, 1
#define ALIGNR_1(a, b) _mm_alignr_epi8(a, b, 15)//alignr a, b, 1
    {
      __m256i a;
      a = ALIGNRX2_1(ref_reg_lo_256, ref_reg_hi_256);
      sum_ref_1_0 = _mm256_add_epi32(sum_ref_1_0, _mm256_sad_epu8(a, src_reg_256));
    }
#define ALIGNRX2_2(a, b) _mm256_alignr_epi8(a, b, 14)
#define ALIGNR_2(a, b) _mm_alignr_epi8(a, b, 14)
    sum_ref_3_5 = _mm256_add_epi32(sum_ref_3_5, _mm256_sad_epu8(ALIGNRX2_2(ref_reg_lo_256, ref_reg_hi_256), src_reg_256));
    tmp = ALIGNR_2(ref_reg_lo, ref_reg_hi);
    sum_ref_7 = _mm_add_epi32(sum_ref_7, _mm_sad_epu8(tmp, src_reg));
/*
    ref_reg_lo = _mm_loadu_si128((const __m128i *)(ref_ptr + 2 * ref_stride));
    ref_reg_hi = _mm_loadu_si128((const __m128i *)(ref_ptr + 2 * ref_stride + 16));
    sum_ref_A = _mm_add_epi32(sum_ref_6, _mm_sad_epu8(ref_reg_lo, src_reg));
    tmp = ALIGNR_1(_mm256_castsi256_si128(ref_reg_lo_256), _mm256_castsi256_si128(ref_regs_hi_256));
    sum_ref_0 = _mm_add_epi32(sum_ref_0, _mm_sad_epu8(tmp, src_reg));
    tmp = ALIGNR_1(ref_reg_lo, ref_reg_hi);
    sum_ref_1 = _mm_add_epi32(sum_ref_1, _mm_sad_epu8(tmp, src_reg));
    sum_ref_4_2 = _mm256_add_epi32(sum_ref_4_2, _mm256_sad_epu8(ref_reg_lo_256, src_reg_256));
    sum_ref_5_3 = _mm256_add_epi32(sum_ref_5_3, _mm256_sad_epu8(ALIGNRX2_2(ref_reg_lo_256, ref_reg_hi_256), src_reg_256));
    tmp = ALIGNR_2(ref_reg_lo, ref_reg_hi);
    sum_ref_B = _mm_add_epi32(sum_ref_7, _mm_sad_epu8(tmp, src_reg));
    ???
*/
    ref_reg_lo_256 = _mm256_permute2x128_si256(ref_reg_lo_256, ref_reg_lo_256, 0x1);
    ref_reg_hi_256 = _mm256_permute2x128_si256(ref_reg_hi_256, ref_reg_hi_256, 0x1);
#endif


#else
    ref_regs_lo[2] = _mm_loadu_si128((const __m128i *)(ref_ptr + 2 * ref_stride));
    ref_regs_hi[2] = _mm_loadu_si128((const __m128i *)(ref_ptr + 2 * ref_stride + 16));
    sum_ref[4] = _mm_add_epi32(sum_ref[4], _mm_sad_epu8(ref_regs_lo[0], src_reg));
    sum_ref[2] = _mm_add_epi32(sum_ref[2], _mm_sad_epu8(ref_regs_lo[1], src_reg));
    sum_ref[6] = _mm_add_epi32(sum_ref[6], _mm_sad_epu8(ref_regs_lo[2], src_reg));
#define ALIGNR_1(a, b) _mm_alignr_epi8(a, b, 15)//alignr a, b, 1
    tmp = ALIGNR_1(ref_regs_lo[0], ref_regs_hi[0]);
    sum_ref[0] = _mm_add_epi32(sum_ref[0], _mm_sad_epu8(tmp, src_reg));
    tmp = ALIGNR_1(ref_regs_lo[2], ref_regs_hi[2]);
    sum_ref[1] = _mm_add_epi32(sum_ref[1], _mm_sad_epu8(tmp, src_reg));
#define ALIGNR_2(a, b) _mm_alignr_epi8(a, b, 14)//alignr a, b, 2
    tmp = ALIGNR_2(ref_regs_lo[0], ref_regs_hi[0]);
    sum_ref[5] = _mm_add_epi32(sum_ref[5], _mm_sad_epu8(tmp, src_reg));
    tmp = ALIGNR_2(ref_regs_lo[1], ref_regs_hi[1]);
    sum_ref[3] = _mm_add_epi32(sum_ref[3], _mm_sad_epu8(tmp, src_reg));
    tmp = ALIGNR_2(ref_regs_lo[2], ref_regs_hi[2]);
    sum_ref[7] = _mm_add_epi32(sum_ref[7], _mm_sad_epu8(tmp, src_reg));
    ref_regs_lo[0] = ref_regs_lo[1];
    ref_regs_lo[1] = ref_regs_lo[2];
    ref_regs_hi[0] = ref_regs_hi[1];
    ref_regs_hi[1] = ref_regs_hi[2];
#endif
    ref_ptr += ref_stride;
    src_ptr += src_stride;
#undef src_reg
  }
IACA_END
#ifdef XYZ
  //sum_ref_0 = _mm_add_epi32(_mm_add_epi32(sum_ref_6, sum_ref_0), _mm_add_epi32(sum_ref_1, sum_ref_7));
  //sum_ref_4_2 = _mm256_add_epi32(sum_ref_4_2, sum_ref_5_3);
  /*sum_ref_0 = _mm_add_epi32(sum_ref_6, sum_ref_7);
  sum_ref_4_2 = _mm256_add_epi32(_mm256_add_epi32(sum_ref_1_0, sum_ref_4_2), sum_ref_5_3);
  sum_ref_4_2 = _mm256_add_epi32(sum_ref_4_2, _mm256_castsi128_si256(sum_ref_0));
  _mm256_storeu_si256((__m256i *)res, sum_ref_4_2);*/
  sum_ref_0 = _mm_add_epi32(sum_ref_6, sum_ref_7);
  sum_ref_2_4 = _mm256_add_epi32(_mm256_add_epi32(sum_ref_1_0, sum_ref_2_4), sum_ref_3_5);
  sum_ref_2_4 = _mm256_add_epi32(sum_ref_2_4, _mm256_castsi128_si256(sum_ref_0));
  _mm256_storeu_si256((__m256i *)res, sum_ref_2_4);
#else
  sum_ref[0] = _mm_add_epi32(_mm_add_epi32(sum_ref[0], sum_ref[1]), _mm_add_epi32(sum_ref[2], sum_ref[3]));
  sum_ref[4] = _mm_add_epi32(_mm_add_epi32(sum_ref[4], sum_ref[5]), _mm_add_epi32(sum_ref[6], sum_ref[7]));
  _mm_storeu_si128((__m128i *)(res), sum_ref[0]);
  _mm_storeu_si128((__m128i *)(res+4), sum_ref[4]);
#endif
}

void vpx_sad16x16x4d_avx2(const uint8_t *src, int src_stride,
                          const uint8_t *const ref[4], int ref_stride,
                          uint32_t res[4]) {
  __m128i src_reg, ref0_reg, ref1_reg, ref2_reg, ref3_reg;
  __m128i sum_ref0, sum_ref1, sum_ref2, sum_ref3;
  __m128i sum_ref0x, sum_ref1x, sum_ref2x, sum_ref3x;
  __m128i sum_mlow, sum_mhigh;
  int i;
  const uint8_t *ref0, *ref1, *ref2, *ref3;

  ref0 = ref[0];
  ref1 = ref[1];
  ref2 = ref[2];
  ref3 = ref[3];
  sum_ref0 = _mm_set1_epi16(0);
  sum_ref1 = _mm_set1_epi16(0);
  sum_ref2 = _mm_set1_epi16(0);
  sum_ref3 = _mm_set1_epi16(0);
  sum_ref0x = _mm_set1_epi16(0);
  sum_ref1x = _mm_set1_epi16(0);
  sum_ref2x = _mm_set1_epi16(0);
  sum_ref3x = _mm_set1_epi16(0);
  for (i = 0; i < 16; i++) {
    // load src and all refs
    src_reg = _mm_loadu_si128((const __m128i *)src);
    ref0_reg = _mm_loadu_si128((const __m128i *)ref0);
    ref1_reg = _mm_loadu_si128((const __m128i *)ref1);
    ref2_reg = _mm_loadu_si128((const __m128i *)ref2);
    ref3_reg = _mm_loadu_si128((const __m128i *)ref3);
    // sum of the absolute differences between every ref-i to src
    ref0_reg = _mm_sad_epu8(ref0_reg, src_reg);
    ref1_reg = _mm_sad_epu8(ref1_reg, src_reg);
    ref2_reg = _mm_sad_epu8(ref2_reg, src_reg);
    ref3_reg = _mm_sad_epu8(ref3_reg, src_reg);
    if ((i&3) == 0/*i == 1 || i == 4 || i == 7 || i == 8 || i == 11 || i == 14*/) {
      sum_ref0x = _mm_add_epi32(sum_ref0x, ref0_reg);
      sum_ref1x = _mm_add_epi32(sum_ref1x, ref1_reg);
      sum_ref2x = _mm_add_epi32(sum_ref2x, ref2_reg);
      sum_ref3x = _mm_add_epi32(sum_ref3x, ref3_reg);
    }
    // sum every ref-i
    sum_ref0 = _mm_add_epi32(sum_ref0, ref0_reg);
    sum_ref1 = _mm_add_epi32(sum_ref1, ref1_reg);
    sum_ref2 = _mm_add_epi32(sum_ref2, ref2_reg);
    sum_ref3 = _mm_add_epi32(sum_ref3, ref3_reg);

    src += src_stride;
    ref0 += ref_stride;
    ref1 += ref_stride;
    ref2 += ref_stride;
    ref3 += ref_stride;
  }
  {
    __m128i sum;
    // in sum_ref-i the result is saved in the first 4 bytes
    // the other 4 bytes are zeroed.
    // sum_ref1 and sum_ref3 are shifted left by 4 bytes
    sum_ref1 = _mm_slli_si128(sum_ref1, 4);
    sum_ref3 = _mm_slli_si128(sum_ref3, 4);

    // merge sum_ref0 and sum_ref1 also sum_ref2 and sum_ref3
    sum_ref0 = _mm_or_si128(sum_ref0, sum_ref1);
    sum_ref2 = _mm_or_si128(sum_ref2, sum_ref3);

    // merge every 64 bit from each sum_ref-i
    sum_mlow = _mm_unpacklo_epi64(sum_ref0, sum_ref2);
    sum_mhigh = _mm_unpackhi_epi64(sum_ref0, sum_ref2);

    // add the low 64 bit to the high 64 bit
    sum = _mm_add_epi32(sum_mlow, sum_mhigh);

    _mm_storeu_si128((__m128i *)(res), sum);
  }

  {
    __m128i sum;
    // in sum_ref-i the result is saved in the first 4 bytes
    // the other 4 bytes are zeroed.
    // sum_ref1 and sum_ref3 are shifted left by 4 bytes
    sum_ref1x = _mm_slli_si128(sum_ref1x, 4);
    sum_ref3x = _mm_slli_si128(sum_ref3x, 4);

    // merge sum_ref0 and sum_ref1 also sum_ref2 and sum_ref3
    sum_ref0x = _mm_or_si128(sum_ref0x, sum_ref1x);
    sum_ref2x = _mm_or_si128(sum_ref2x, sum_ref3x);

    // merge every 64 bit from each sum_ref-i
    sum_mlow = _mm_unpacklo_epi64(sum_ref0x, sum_ref2x);
    sum_mhigh = _mm_unpackhi_epi64(sum_ref0x, sum_ref2x);

    // add the low 64 bit to the high 64 bit
    sum = _mm_add_epi32(sum_mlow, sum_mhigh);
    
    other = sum;
  }
}

void vpx_sad32x32x4d_avx2(const uint8_t *src, int src_stride,
                          const uint8_t *const ref[4], int ref_stride,
                          uint32_t res[4]) {
  __m256i src_reg, ref0_reg, ref1_reg, ref2_reg, ref3_reg;
  __m256i sum_ref0, sum_ref1, sum_ref2, sum_ref3;
  __m256i sum_mlow, sum_mhigh;
  int i;
  const uint8_t *ref0, *ref1, *ref2, *ref3;

  ref0 = ref[0];
  ref1 = ref[1];
  ref2 = ref[2];
  ref3 = ref[3];
  sum_ref0 = _mm256_set1_epi16(0);
  sum_ref1 = _mm256_set1_epi16(0);
  sum_ref2 = _mm256_set1_epi16(0);
  sum_ref3 = _mm256_set1_epi16(0);
  for (i = 0; i < 32; i++) {
    // load src and all refs
    src_reg = _mm256_loadu_si256((const __m256i *)src);
    ref0_reg = _mm256_loadu_si256((const __m256i *)ref0);
    ref1_reg = _mm256_loadu_si256((const __m256i *)ref1);
    ref2_reg = _mm256_loadu_si256((const __m256i *)ref2);
    ref3_reg = _mm256_loadu_si256((const __m256i *)ref3);
    // sum of the absolute differences between every ref-i to src
    ref0_reg = _mm256_sad_epu8(ref0_reg, src_reg);
    ref1_reg = _mm256_sad_epu8(ref1_reg, src_reg);
    ref2_reg = _mm256_sad_epu8(ref2_reg, src_reg);
    ref3_reg = _mm256_sad_epu8(ref3_reg, src_reg);
    // sum every ref-i
    sum_ref0 = _mm256_add_epi32(sum_ref0, ref0_reg);
    sum_ref1 = _mm256_add_epi32(sum_ref1, ref1_reg);
    sum_ref2 = _mm256_add_epi32(sum_ref2, ref2_reg);
    sum_ref3 = _mm256_add_epi32(sum_ref3, ref3_reg);

    src += src_stride;
    ref0 += ref_stride;
    ref1 += ref_stride;
    ref2 += ref_stride;
    ref3 += ref_stride;
  }
  {
    __m128i sum;
    // in sum_ref-i the result is saved in the first 4 bytes
    // the other 4 bytes are zeroed.
    // sum_ref1 and sum_ref3 are shifted left by 4 bytes
    sum_ref1 = _mm256_slli_si256(sum_ref1, 4);
    sum_ref3 = _mm256_slli_si256(sum_ref3, 4);

    // merge sum_ref0 and sum_ref1 also sum_ref2 and sum_ref3
    sum_ref0 = _mm256_or_si256(sum_ref0, sum_ref1);
    sum_ref2 = _mm256_or_si256(sum_ref2, sum_ref3);

    // merge every 64 bit from each sum_ref-i
    sum_mlow = _mm256_unpacklo_epi64(sum_ref0, sum_ref2);
    sum_mhigh = _mm256_unpackhi_epi64(sum_ref0, sum_ref2);

    // add the low 64 bit to the high 64 bit
    sum_mlow = _mm256_add_epi32(sum_mlow, sum_mhigh);

    // add the low 128 bit to the high 128 bit
    sum = _mm_add_epi32(_mm256_castsi256_si128(sum_mlow),
                        _mm256_extractf128_si256(sum_mlow, 1));

    _mm_storeu_si128((__m128i *)(res), sum);
  }
}

void vpx_sad64x64x4d_avx2(const uint8_t *src, int src_stride,
                          const uint8_t *const ref[4], int ref_stride,
                          uint32_t res[4]) {
  __m256i src_reg, srcnext_reg, ref0_reg, ref0next_reg;
  __m256i ref1_reg, ref1next_reg, ref2_reg, ref2next_reg;
  __m256i ref3_reg, ref3next_reg;
  __m256i sum_ref0, sum_ref1, sum_ref2, sum_ref3;
  __m256i sum_mlow, sum_mhigh;
  int i;
  const uint8_t *ref0, *ref1, *ref2, *ref3;

  ref0 = ref[0];
  ref1 = ref[1];
  ref2 = ref[2];
  ref3 = ref[3];
  sum_ref0 = _mm256_set1_epi16(0);
  sum_ref1 = _mm256_set1_epi16(0);
  sum_ref2 = _mm256_set1_epi16(0);
  sum_ref3 = _mm256_set1_epi16(0);
  for (i = 0; i < 64; i++) {
    // load 64 bytes from src and all refs
    src_reg = _mm256_loadu_si256((const __m256i *)src);
    srcnext_reg = _mm256_loadu_si256((const __m256i *)(src + 32));
    ref0_reg = _mm256_loadu_si256((const __m256i *)ref0);
    ref0next_reg = _mm256_loadu_si256((const __m256i *)(ref0 + 32));
    ref1_reg = _mm256_loadu_si256((const __m256i *)ref1);
    ref1next_reg = _mm256_loadu_si256((const __m256i *)(ref1 + 32));
    ref2_reg = _mm256_loadu_si256((const __m256i *)ref2);
    ref2next_reg = _mm256_loadu_si256((const __m256i *)(ref2 + 32));
    ref3_reg = _mm256_loadu_si256((const __m256i *)ref3);
    ref3next_reg = _mm256_loadu_si256((const __m256i *)(ref3 + 32));
    // sum of the absolute differences between every ref-i to src
    ref0_reg = _mm256_sad_epu8(ref0_reg, src_reg);
    ref1_reg = _mm256_sad_epu8(ref1_reg, src_reg);
    ref2_reg = _mm256_sad_epu8(ref2_reg, src_reg);
    ref3_reg = _mm256_sad_epu8(ref3_reg, src_reg);
    ref0next_reg = _mm256_sad_epu8(ref0next_reg, srcnext_reg);
    ref1next_reg = _mm256_sad_epu8(ref1next_reg, srcnext_reg);
    ref2next_reg = _mm256_sad_epu8(ref2next_reg, srcnext_reg);
    ref3next_reg = _mm256_sad_epu8(ref3next_reg, srcnext_reg);

    // sum every ref-i
    sum_ref0 = _mm256_add_epi32(sum_ref0, ref0_reg);
    sum_ref1 = _mm256_add_epi32(sum_ref1, ref1_reg);
    sum_ref2 = _mm256_add_epi32(sum_ref2, ref2_reg);
    sum_ref3 = _mm256_add_epi32(sum_ref3, ref3_reg);
    sum_ref0 = _mm256_add_epi32(sum_ref0, ref0next_reg);
    sum_ref1 = _mm256_add_epi32(sum_ref1, ref1next_reg);
    sum_ref2 = _mm256_add_epi32(sum_ref2, ref2next_reg);
    sum_ref3 = _mm256_add_epi32(sum_ref3, ref3next_reg);
    src += src_stride;
    ref0 += ref_stride;
    ref1 += ref_stride;
    ref2 += ref_stride;
    ref3 += ref_stride;
  }
  {
    __m128i sum;

    // in sum_ref-i the result is saved in the first 4 bytes
    // the other 4 bytes are zeroed.
    // sum_ref1 and sum_ref3 are shifted left by 4 bytes
    sum_ref1 = _mm256_slli_si256(sum_ref1, 4);
    sum_ref3 = _mm256_slli_si256(sum_ref3, 4);

    // merge sum_ref0 and sum_ref1 also sum_ref2 and sum_ref3
    sum_ref0 = _mm256_or_si256(sum_ref0, sum_ref1);
    sum_ref2 = _mm256_or_si256(sum_ref2, sum_ref3);

    // merge every 64 bit from each sum_ref-i
    sum_mlow = _mm256_unpacklo_epi64(sum_ref0, sum_ref2);
    sum_mhigh = _mm256_unpackhi_epi64(sum_ref0, sum_ref2);

    // add the low 64 bit to the high 64 bit
    sum_mlow = _mm256_add_epi32(sum_mlow, sum_mhigh);

    // add the low 128 bit to the high 128 bit
    sum = _mm_add_epi32(_mm256_castsi256_si128(sum_mlow),
                        _mm256_extractf128_si256(sum_mlow, 1));

    _mm_storeu_si128((__m128i *)(res), sum);
  }
}
