/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <immintrin.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/x86/convolve.h"
#include "vpx_ports/mem.h"

// filters for 16_h8 and 16_v8
DECLARE_ALIGNED(32, static const uint8_t, filt1_global_avx2[32]) = {
  0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,
  0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
};

DECLARE_ALIGNED(32, static const uint8_t, filt2_global_avx2[32]) = {
  2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10,
  2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10
};

DECLARE_ALIGNED(32, static const uint8_t, filt3_global_avx2[32]) = {
  4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12,
  4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12
};

DECLARE_ALIGNED(32, static const uint8_t, filt4_global_avx2[32]) = {
  6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14,
  6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14
};

#if defined(__clang__)
#if (__clang_major__ > 0 && __clang_major__ < 3) ||            \
    (__clang_major__ == 3 && __clang_minor__ <= 3) ||          \
    (defined(__APPLE__) && defined(__apple_build_version__) && \
     ((__clang_major__ == 4 && __clang_minor__ <= 2) ||        \
      (__clang_major__ == 5 && __clang_minor__ == 0)))
#define MM256_BROADCASTSI128_SI256(x) \
  _mm_broadcastsi128_si256((__m128i const *)&(x))
#else  // clang > 3.3, and not 5.0 on macosx.
#define MM256_BROADCASTSI128_SI256(x) _mm256_broadcastsi128_si256(x)
#endif  // clang <= 3.3
#elif defined(__GNUC__)
#if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ <= 6)
#define MM256_BROADCASTSI128_SI256(x) \
  _mm_broadcastsi128_si256((__m128i const *)&(x))
#elif __GNUC__ == 4 && __GNUC_MINOR__ == 7
#define MM256_BROADCASTSI128_SI256(x) _mm_broadcastsi128_si256(x)
#else  // gcc > 4.7
#define MM256_BROADCASTSI128_SI256(x) _mm256_broadcastsi128_si256(x)
#endif  // gcc <= 4.6
#else   // !(gcc || clang)
#define MM256_BROADCASTSI128_SI256(x) _mm256_broadcastsi128_si256(x)
#endif  // __clang__

static INLINE void shuffle_filter_avx2(const int16_t *const filter,
                                       __m256i *const f) {
  const __m256i f_values =
      MM256_BROADCASTSI128_SI256(_mm_load_si128((const __m128i *)filter));
  // pack and duplicate the filter values
  f[0] = _mm256_shuffle_epi8(f_values, _mm256_set1_epi16(0x0200u));
  f[1] = _mm256_shuffle_epi8(f_values, _mm256_set1_epi16(0x0604u));
  f[2] = _mm256_shuffle_epi8(f_values, _mm256_set1_epi16(0x0a08u));
  f[3] = _mm256_shuffle_epi8(f_values, _mm256_set1_epi16(0x0e0cu));
}

static INLINE __m256i convolve8_16_avx2(const __m256i *const s,
                                        const __m256i *const f) {
  // multiply 2 adjacent elements with the filter and add the result
  const __m256i k_64 = _mm256_set1_epi16(1 << 6);
  const __m256i x0 = _mm256_maddubs_epi16(s[0], f[0]);
  const __m256i x1 = _mm256_maddubs_epi16(s[1], f[1]);
  const __m256i x2 = _mm256_maddubs_epi16(s[2], f[2]);
  const __m256i x3 = _mm256_maddubs_epi16(s[3], f[3]);
  // add and saturate the results together
  const __m256i min_x2x1 = _mm256_min_epi16(x2, x1);
  const __m256i max_x2x1 = _mm256_max_epi16(x2, x1);
  __m256i temp = _mm256_adds_epi16(x0, x3);
  temp = _mm256_adds_epi16(temp, min_x2x1);
  temp = _mm256_adds_epi16(temp, max_x2x1);
  // round and shift by 7 bit each 16 bit
  temp = _mm256_adds_epi16(temp, k_64);
  temp = _mm256_srai_epi16(temp, 7);
  return temp;
}

static INLINE __m128i convolve8_8_avx2(const __m256i *const s,
                                        const __m256i *const f) {
  // multiply 2 adjacent elements with the filter and add the result
  const __m128i k_64 = _mm_set1_epi16(1 << 6);
  const __m128i x0 = _mm_maddubs_epi16(_mm256_castsi256_si128(s[0]), _mm256_castsi256_si128(f[0]));
  const __m128i x1 = _mm_maddubs_epi16(_mm256_castsi256_si128(s[1]), _mm256_castsi256_si128(f[1]));
  const __m128i x2 = _mm_maddubs_epi16(_mm256_castsi256_si128(s[2]), _mm256_castsi256_si128(f[2]));
  const __m128i x3 = _mm_maddubs_epi16(_mm256_castsi256_si128(s[3]), _mm256_castsi256_si128(f[3]));
  // add and saturate the results together
  const __m128i min_x2x1 = _mm_min_epi16(x2, x1);
  const __m128i max_x2x1 = _mm_max_epi16(x2, x1);
  __m128i temp = _mm_adds_epi16(x0, x3);
  temp = _mm_adds_epi16(temp, min_x2x1);
  temp = _mm_adds_epi16(temp, max_x2x1);
  // round and shift by 7 bit each 16 bit
  temp = _mm_adds_epi16(temp, k_64);
  temp = _mm_srai_epi16(temp, 7);
  return temp;
}

static INLINE void vpx_filter_block1d16_h8_X_avx2(
    const uint8_t *src_ptr, ptrdiff_t src_pixels_per_line, uint8_t *output_ptr,
    ptrdiff_t output_pitch, uint32_t output_height, const int16_t *filter,
    int avg) {
  __m256i f[4], filt[4], s[4];
  __m128i outReg;
  __m256i srcRegFilt32b1_1, srcRegFilt32b2_1;
  __m256i srcReg32b1, srcReg32b2;
  unsigned int i;
  ptrdiff_t src_stride, dst_stride;

  shuffle_filter_avx2(filter, f);

  filt[0] = _mm256_load_si256((__m256i const *)filt1_global_avx2);
  filt[1] = _mm256_load_si256((__m256i const *)filt2_global_avx2);
  filt[2] = _mm256_load_si256((__m256i const *)filt3_global_avx2);
  filt[3] = _mm256_load_si256((__m256i const *)filt4_global_avx2);

  // multiple the size of the source and destination stride by two
  src_stride = src_pixels_per_line << 1;
  dst_stride = output_pitch << 1;
  for (i = output_height; i > 1; i -= 2) {
    // load the 2 strides of source
    srcReg32b1 =
        _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(src_ptr - 3)));
    srcReg32b1 = _mm256_inserti128_si256(
        srcReg32b1,
        _mm_loadu_si128((const __m128i *)(src_ptr + src_pixels_per_line - 3)),
        1);

    // filter the source buffer
    s[0] = _mm256_shuffle_epi8(srcReg32b1, filt[0]);
    s[1] = _mm256_shuffle_epi8(srcReg32b1, filt[1]);
    s[2] = _mm256_shuffle_epi8(srcReg32b1, filt[2]);
    s[3] = _mm256_shuffle_epi8(srcReg32b1, filt[3]);
    srcRegFilt32b1_1 = convolve8_16_avx2(s, f);

    // reading 2 strides of the next 16 bytes
    // (part of it was being read by earlier read)
    srcReg32b2 =
        _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(src_ptr + 5)));
    srcReg32b2 = _mm256_inserti128_si256(
        srcReg32b2,
        _mm_loadu_si128((const __m128i *)(src_ptr + src_pixels_per_line + 5)),
        1);

    // filter the source buffer
    s[0] = _mm256_shuffle_epi8(srcReg32b2, filt[0]);
    s[1] = _mm256_shuffle_epi8(srcReg32b2, filt[1]);
    s[2] = _mm256_shuffle_epi8(srcReg32b2, filt[2]);
    s[3] = _mm256_shuffle_epi8(srcReg32b2, filt[3]);
    srcRegFilt32b2_1 = convolve8_16_avx2(s, f);

    // shrink to 8 bit each 16 bits, the first lane contain the first
    // convolve result and the second lane contain the second convolve
    // result
    srcRegFilt32b1_1 = _mm256_packus_epi16(srcRegFilt32b1_1, srcRegFilt32b2_1);

    src_ptr += src_stride;

    // average if necessary
    outReg = _mm256_castsi256_si128(srcRegFilt32b1_1);
    if (avg) {
      outReg = _mm_avg_epu8(outReg, _mm_load_si128((__m128i *)output_ptr));
    }

    // save 16 bytes
    _mm_store_si128((__m128i *)output_ptr, outReg);

    // average if necessary
    outReg = _mm256_extractf128_si256(srcRegFilt32b1_1, 1);
    if (avg) {
      outReg = _mm_avg_epu8(
          outReg, _mm_load_si128((__m128i *)(output_ptr + output_pitch)));
    }

    // save the next 16 bits
    _mm_store_si128((__m128i *)(output_ptr + output_pitch), outReg);

    output_ptr += dst_stride;
  }

  // if the number of strides is odd.
  // process only 16 bytes
  if (i > 0) {
    __m128i srcReg1, srcReg2, srcRegFilt1_1, srcRegFilt2_1;

    srcReg1 = _mm_loadu_si128((const __m128i *)(src_ptr - 3));

    // filter the source buffer
    s[0] = _mm256_castsi128_si256(_mm_shuffle_epi8(srcReg1, _mm256_castsi256_si128(filt[0])));
    s[1] = _mm256_castsi128_si256(_mm_shuffle_epi8(srcReg1, _mm256_castsi256_si128(filt[1])));
    s[2] = _mm256_castsi128_si256(_mm_shuffle_epi8(srcReg1, _mm256_castsi256_si128(filt[2])));
    s[3] = _mm256_castsi128_si256(_mm_shuffle_epi8(srcReg1, _mm256_castsi256_si128(filt[3])));

    srcRegFilt1_1 = convolve8_8_avx2(s, f);

    // reading the next 16 bytes
    // (part of it was being read by earlier read)
    srcReg2 = _mm_loadu_si128((const __m128i *)(src_ptr + 5));

    // filter the source buffer
    s[0] = _mm256_castsi128_si256(_mm_shuffle_epi8(srcReg2, _mm256_castsi256_si128(filt[0])));
    s[1] = _mm256_castsi128_si256(_mm_shuffle_epi8(srcReg2, _mm256_castsi256_si128(filt[1])));
    s[2] = _mm256_castsi128_si256(_mm_shuffle_epi8(srcReg2, _mm256_castsi256_si128(filt[2])));
    s[3] = _mm256_castsi128_si256(_mm_shuffle_epi8(srcReg2, _mm256_castsi256_si128(filt[3])));

    srcRegFilt2_1 = convolve8_8_avx2(s, f);

    // shrink to 8 bit each 16 bits, the first lane contain the first
    // convolve result and the second lane contain the second convolve
    // result
    outReg = _mm_packus_epi16(srcRegFilt1_1, srcRegFilt2_1);

    // average if necessary
    if (avg) {
      outReg = _mm_avg_epu8(outReg, _mm_load_si128((__m128i *)output_ptr));
    }

    // save 16 bytes
    _mm_store_si128((__m128i *)output_ptr, outReg);
  }
}

static void vpx_filter_block1d16_h8_avx2(
    const uint8_t *src_ptr, ptrdiff_t src_stride, uint8_t *output_ptr,
    ptrdiff_t dst_stride, uint32_t output_height, const int16_t *filter) {
  vpx_filter_block1d16_h8_X_avx2(src_ptr, src_stride, output_ptr, dst_stride,
                                 output_height, filter, 0);
}

static void vpx_filter_block1d16_h8_avg_avx2(
    const uint8_t *src_ptr, ptrdiff_t src_stride, uint8_t *output_ptr,
    ptrdiff_t dst_stride, uint32_t output_height, const int16_t *filter) {
  vpx_filter_block1d16_h8_X_avx2(src_ptr, src_stride, output_ptr, dst_stride,
                                 output_height, filter, 1);
}

static INLINE void vpx_filter_block1d16_v8_X_avx2(
    const uint8_t *src_ptr, ptrdiff_t src_pitch, uint8_t *output_ptr,
    ptrdiff_t out_pitch, uint32_t output_height, const int16_t *filter,
    int avg) {
  __m128i filtersReg, outReg;
  __m256i addFilterReg64;
  __m256i srcReg32b1, srcReg32b2, srcReg32b3, srcReg32b4, srcReg32b5;
  __m256i srcReg32b6, srcReg32b7, srcReg32b8, srcReg32b9, srcReg32b10;
  __m256i srcReg32b11, srcReg32b12, filtersReg32;
  __m256i firstFilters, secondFilters, thirdFilters, forthFilters;
  unsigned int i;
  ptrdiff_t src_stride, dst_stride;

  // create a register with 0,64,0,64,0,64,0,64,0,64,0,64,0,64,0,64
  addFilterReg64 = _mm256_set1_epi32((int)0x0400040u);
  filtersReg = _mm_loadu_si128((const __m128i *)filter);
  // converting the 16 bit (short) to  8 bit (byte) and have the
  // same data in both lanes of 128 bit register.
  filtersReg = _mm_packs_epi16(filtersReg, filtersReg);
  // have the same data in both lanes of a 256 bit register
  filtersReg32 = MM256_BROADCASTSI128_SI256(filtersReg);

  // duplicate only the first 16 bits (first and second byte)
  // across 256 bit register
  firstFilters = _mm256_shuffle_epi8(filtersReg32, _mm256_set1_epi16(0x100u));
  // duplicate only the second 16 bits (third and forth byte)
  // across 256 bit register
  secondFilters = _mm256_shuffle_epi8(filtersReg32, _mm256_set1_epi16(0x302u));
  // duplicate only the third 16 bits (fifth and sixth byte)
  // across 256 bit register
  thirdFilters = _mm256_shuffle_epi8(filtersReg32, _mm256_set1_epi16(0x504u));
  // duplicate only the forth 16 bits (seventh and eighth byte)
  // across 256 bit register
  forthFilters = _mm256_shuffle_epi8(filtersReg32, _mm256_set1_epi16(0x706u));

  // multiple the size of the source and destination stride by two
  src_stride = src_pitch << 1;
  dst_stride = out_pitch << 1;

  // load 16 bytes 7 times in stride of src_pitch
  srcReg32b1 =
      _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(src_ptr)));
  srcReg32b2 = _mm256_castsi128_si256(
      _mm_loadu_si128((const __m128i *)(src_ptr + src_pitch)));
  srcReg32b3 = _mm256_castsi128_si256(
      _mm_loadu_si128((const __m128i *)(src_ptr + src_pitch * 2)));
  srcReg32b4 = _mm256_castsi128_si256(
      _mm_loadu_si128((const __m128i *)(src_ptr + src_pitch * 3)));
  srcReg32b5 = _mm256_castsi128_si256(
      _mm_loadu_si128((const __m128i *)(src_ptr + src_pitch * 4)));
  srcReg32b6 = _mm256_castsi128_si256(
      _mm_loadu_si128((const __m128i *)(src_ptr + src_pitch * 5)));
  srcReg32b7 = _mm256_castsi128_si256(
      _mm_loadu_si128((const __m128i *)(src_ptr + src_pitch * 6)));

  // have each consecutive loads on the same 256 register
  srcReg32b1 = _mm256_inserti128_si256(srcReg32b1,
                                       _mm256_castsi256_si128(srcReg32b2), 1);
  srcReg32b2 = _mm256_inserti128_si256(srcReg32b2,
                                       _mm256_castsi256_si128(srcReg32b3), 1);
  srcReg32b3 = _mm256_inserti128_si256(srcReg32b3,
                                       _mm256_castsi256_si128(srcReg32b4), 1);
  srcReg32b4 = _mm256_inserti128_si256(srcReg32b4,
                                       _mm256_castsi256_si128(srcReg32b5), 1);
  srcReg32b5 = _mm256_inserti128_si256(srcReg32b5,
                                       _mm256_castsi256_si128(srcReg32b6), 1);
  srcReg32b6 = _mm256_inserti128_si256(srcReg32b6,
                                       _mm256_castsi256_si128(srcReg32b7), 1);

  // merge every two consecutive registers except the last one
  srcReg32b10 = _mm256_unpacklo_epi8(srcReg32b1, srcReg32b2);
  srcReg32b1 = _mm256_unpackhi_epi8(srcReg32b1, srcReg32b2);

  // save
  srcReg32b11 = _mm256_unpacklo_epi8(srcReg32b3, srcReg32b4);

  // save
  srcReg32b3 = _mm256_unpackhi_epi8(srcReg32b3, srcReg32b4);

  // save
  srcReg32b2 = _mm256_unpacklo_epi8(srcReg32b5, srcReg32b6);

  // save
  srcReg32b5 = _mm256_unpackhi_epi8(srcReg32b5, srcReg32b6);

  for (i = output_height; i > 1; i -= 2) {
    // load the last 2 loads of 16 bytes and have every two
    // consecutive loads in the same 256 bit register
    srcReg32b8 = _mm256_castsi128_si256(
        _mm_loadu_si128((const __m128i *)(src_ptr + src_pitch * 7)));
    srcReg32b7 = _mm256_inserti128_si256(srcReg32b7,
                                         _mm256_castsi256_si128(srcReg32b8), 1);
    srcReg32b9 = _mm256_castsi128_si256(
        _mm_loadu_si128((const __m128i *)(src_ptr + src_pitch * 8)));
    srcReg32b8 = _mm256_inserti128_si256(srcReg32b8,
                                         _mm256_castsi256_si128(srcReg32b9), 1);

    // merge every two consecutive registers
    // save
    srcReg32b4 = _mm256_unpacklo_epi8(srcReg32b7, srcReg32b8);
    srcReg32b7 = _mm256_unpackhi_epi8(srcReg32b7, srcReg32b8);

    // multiply 2 adjacent elements with the filter and add the result
    srcReg32b10 = _mm256_maddubs_epi16(srcReg32b10, firstFilters);
    srcReg32b6 = _mm256_maddubs_epi16(srcReg32b4, forthFilters);

    // add and saturate the results together
    srcReg32b10 = _mm256_adds_epi16(srcReg32b10, srcReg32b6);

    // multiply 2 adjacent elements with the filter and add the result
    srcReg32b8 = _mm256_maddubs_epi16(srcReg32b11, secondFilters);
    srcReg32b12 = _mm256_maddubs_epi16(srcReg32b2, thirdFilters);

    // add and saturate the results together
    srcReg32b10 = _mm256_adds_epi16(srcReg32b10,
                                    _mm256_min_epi16(srcReg32b8, srcReg32b12));
    srcReg32b10 = _mm256_adds_epi16(srcReg32b10,
                                    _mm256_max_epi16(srcReg32b8, srcReg32b12));

    // multiply 2 adjacent elements with the filter and add the result
    srcReg32b1 = _mm256_maddubs_epi16(srcReg32b1, firstFilters);
    srcReg32b6 = _mm256_maddubs_epi16(srcReg32b7, forthFilters);

    srcReg32b1 = _mm256_adds_epi16(srcReg32b1, srcReg32b6);

    // multiply 2 adjacent elements with the filter and add the result
    srcReg32b8 = _mm256_maddubs_epi16(srcReg32b3, secondFilters);
    srcReg32b12 = _mm256_maddubs_epi16(srcReg32b5, thirdFilters);

    // add and saturate the results together
    srcReg32b1 = _mm256_adds_epi16(srcReg32b1,
                                   _mm256_min_epi16(srcReg32b8, srcReg32b12));
    srcReg32b1 = _mm256_adds_epi16(srcReg32b1,
                                   _mm256_max_epi16(srcReg32b8, srcReg32b12));

    srcReg32b10 = _mm256_adds_epi16(srcReg32b10, addFilterReg64);
    srcReg32b1 = _mm256_adds_epi16(srcReg32b1, addFilterReg64);

    // shift by 7 bit each 16 bit
    srcReg32b10 = _mm256_srai_epi16(srcReg32b10, 7);
    srcReg32b1 = _mm256_srai_epi16(srcReg32b1, 7);

    // shrink to 8 bit each 16 bits, the first lane contain the first
    // convolve result and the second lane contain the second convolve
    // result
    srcReg32b1 = _mm256_packus_epi16(srcReg32b10, srcReg32b1);

    src_ptr += src_stride;

    // average if necessary
    outReg = _mm256_castsi256_si128(srcReg32b1);
    if (avg) {
      outReg = _mm_avg_epu8(outReg, _mm_load_si128((__m128i *)output_ptr));
    }

    // save 16 bytes
    _mm_store_si128((__m128i *)output_ptr, outReg);

    // average if necessary
    outReg = _mm256_extractf128_si256(srcReg32b1, 1);
    if (avg) {
      outReg = _mm_avg_epu8(
          outReg, _mm_load_si128((__m128i *)(output_ptr + out_pitch)));
    }

    // save the next 16 bits
    _mm_store_si128((__m128i *)(output_ptr + out_pitch), outReg);

    output_ptr += dst_stride;

    // save part of the registers for next strides
    srcReg32b10 = srcReg32b11;
    srcReg32b1 = srcReg32b3;
    srcReg32b11 = srcReg32b2;
    srcReg32b3 = srcReg32b5;
    srcReg32b2 = srcReg32b4;
    srcReg32b5 = srcReg32b7;
    srcReg32b7 = srcReg32b9;
  }
  if (i > 0) {
    __m128i srcRegFilt1, srcRegFilt3, srcRegFilt4, srcRegFilt5;
    __m128i srcRegFilt6, srcRegFilt7, srcRegFilt8;
    // load the last 16 bytes
    srcRegFilt8 = _mm_loadu_si128((const __m128i *)(src_ptr + src_pitch * 7));

    // merge the last 2 results together
    srcRegFilt4 =
        _mm_unpacklo_epi8(_mm256_castsi256_si128(srcReg32b7), srcRegFilt8);
    srcRegFilt7 =
        _mm_unpackhi_epi8(_mm256_castsi256_si128(srcReg32b7), srcRegFilt8);

    // multiply 2 adjacent elements with the filter and add the result
    srcRegFilt1 = _mm_maddubs_epi16(_mm256_castsi256_si128(srcReg32b10),
                                    _mm256_castsi256_si128(firstFilters));
    srcRegFilt4 =
        _mm_maddubs_epi16(srcRegFilt4, _mm256_castsi256_si128(forthFilters));
    srcRegFilt3 = _mm_maddubs_epi16(_mm256_castsi256_si128(srcReg32b1),
                                    _mm256_castsi256_si128(firstFilters));
    srcRegFilt7 =
        _mm_maddubs_epi16(srcRegFilt7, _mm256_castsi256_si128(forthFilters));

    // add and saturate the results together
    srcRegFilt1 = _mm_adds_epi16(srcRegFilt1, srcRegFilt4);
    srcRegFilt3 = _mm_adds_epi16(srcRegFilt3, srcRegFilt7);

    // multiply 2 adjacent elements with the filter and add the result
    srcRegFilt4 = _mm_maddubs_epi16(_mm256_castsi256_si128(srcReg32b11),
                                    _mm256_castsi256_si128(secondFilters));
    srcRegFilt5 = _mm_maddubs_epi16(_mm256_castsi256_si128(srcReg32b3),
                                    _mm256_castsi256_si128(secondFilters));

    // multiply 2 adjacent elements with the filter and add the result
    srcRegFilt6 = _mm_maddubs_epi16(_mm256_castsi256_si128(srcReg32b2),
                                    _mm256_castsi256_si128(thirdFilters));
    srcRegFilt7 = _mm_maddubs_epi16(_mm256_castsi256_si128(srcReg32b5),
                                    _mm256_castsi256_si128(thirdFilters));

    // add and saturate the results together
    srcRegFilt1 =
        _mm_adds_epi16(srcRegFilt1, _mm_min_epi16(srcRegFilt4, srcRegFilt6));
    srcRegFilt3 =
        _mm_adds_epi16(srcRegFilt3, _mm_min_epi16(srcRegFilt5, srcRegFilt7));

    // add and saturate the results together
    srcRegFilt1 =
        _mm_adds_epi16(srcRegFilt1, _mm_max_epi16(srcRegFilt4, srcRegFilt6));
    srcRegFilt3 =
        _mm_adds_epi16(srcRegFilt3, _mm_max_epi16(srcRegFilt5, srcRegFilt7));

    srcRegFilt1 =
        _mm_adds_epi16(srcRegFilt1, _mm256_castsi256_si128(addFilterReg64));
    srcRegFilt3 =
        _mm_adds_epi16(srcRegFilt3, _mm256_castsi256_si128(addFilterReg64));

    // shift by 7 bit each 16 bit
    srcRegFilt1 = _mm_srai_epi16(srcRegFilt1, 7);
    srcRegFilt3 = _mm_srai_epi16(srcRegFilt3, 7);

    // shrink to 8 bit each 16 bits, the first lane contain the first
    // convolve result and the second lane contain the second convolve
    // result
    outReg = _mm_packus_epi16(srcRegFilt1, srcRegFilt3);

    // average if necessary
    if (avg) {
      outReg = _mm_avg_epu8(outReg, _mm_load_si128((__m128i *)output_ptr));
    }

    // save 16 bytes
    _mm_store_si128((__m128i *)output_ptr, outReg);
  }
}

static void vpx_filter_block1d16_v8_avx2(const uint8_t *src_ptr,
                                         ptrdiff_t src_stride, uint8_t *dst_ptr,
                                         ptrdiff_t dst_stride, uint32_t height,
                                         const int16_t *filter) {
  vpx_filter_block1d16_v8_X_avx2(src_ptr, src_stride, dst_ptr, dst_stride,
                                 height, filter, 0);
}

static void vpx_filter_block1d16_v8_avg_avx2(
    const uint8_t *src_ptr, ptrdiff_t src_stride, uint8_t *dst_ptr,
    ptrdiff_t dst_stride, uint32_t height, const int16_t *filter) {
  vpx_filter_block1d16_v8_X_avx2(src_ptr, src_stride, dst_ptr, dst_stride,
                                 height, filter, 1);
}

#if HAVE_AVX2 && HAVE_SSSE3
filter8_1dfunction vpx_filter_block1d4_v8_ssse3;
#if ARCH_X86_64
filter8_1dfunction vpx_filter_block1d8_v8_intrin_ssse3;
filter8_1dfunction vpx_filter_block1d8_h8_intrin_ssse3;
filter8_1dfunction vpx_filter_block1d4_h8_intrin_ssse3;
#define vpx_filter_block1d8_v8_avx2 vpx_filter_block1d8_v8_intrin_ssse3
#define vpx_filter_block1d8_h8_avx2 vpx_filter_block1d8_h8_intrin_ssse3
#define vpx_filter_block1d4_h8_avx2 vpx_filter_block1d4_h8_intrin_ssse3
#else  // ARCH_X86
filter8_1dfunction vpx_filter_block1d8_v8_ssse3;
filter8_1dfunction vpx_filter_block1d8_h8_ssse3;
filter8_1dfunction vpx_filter_block1d4_h8_ssse3;
#define vpx_filter_block1d8_v8_avx2 vpx_filter_block1d8_v8_ssse3
#define vpx_filter_block1d8_h8_avx2 vpx_filter_block1d8_h8_ssse3
#define vpx_filter_block1d4_h8_avx2 vpx_filter_block1d4_h8_ssse3
#endif  // ARCH_X86_64
filter8_1dfunction vpx_filter_block1d8_v8_avg_ssse3;
filter8_1dfunction vpx_filter_block1d8_h8_avg_ssse3;
filter8_1dfunction vpx_filter_block1d4_v8_avg_ssse3;
filter8_1dfunction vpx_filter_block1d4_h8_avg_ssse3;
#define vpx_filter_block1d8_v8_avg_avx2 vpx_filter_block1d8_v8_avg_ssse3
#define vpx_filter_block1d8_h8_avg_avx2 vpx_filter_block1d8_h8_avg_ssse3
#define vpx_filter_block1d4_v8_avg_avx2 vpx_filter_block1d4_v8_avg_ssse3
#define vpx_filter_block1d4_h8_avg_avx2 vpx_filter_block1d4_h8_avg_ssse3
filter8_1dfunction vpx_filter_block1d16_v2_ssse3;
filter8_1dfunction vpx_filter_block1d16_h2_ssse3;
filter8_1dfunction vpx_filter_block1d8_v2_ssse3;
filter8_1dfunction vpx_filter_block1d8_h2_ssse3;
filter8_1dfunction vpx_filter_block1d4_v2_ssse3;
filter8_1dfunction vpx_filter_block1d4_h2_ssse3;
#define vpx_filter_block1d4_v8_avx2 vpx_filter_block1d4_v8_ssse3
#define vpx_filter_block1d16_v2_avx2 vpx_filter_block1d16_v2_ssse3
#define vpx_filter_block1d16_h2_avx2 vpx_filter_block1d16_h2_ssse3
#define vpx_filter_block1d8_v2_avx2 vpx_filter_block1d8_v2_ssse3
#define vpx_filter_block1d8_h2_avx2 vpx_filter_block1d8_h2_ssse3
#define vpx_filter_block1d4_v2_avx2 vpx_filter_block1d4_v2_ssse3
#define vpx_filter_block1d4_h2_avx2 vpx_filter_block1d4_h2_ssse3
filter8_1dfunction vpx_filter_block1d16_v2_avg_ssse3;
filter8_1dfunction vpx_filter_block1d16_h2_avg_ssse3;
filter8_1dfunction vpx_filter_block1d8_v2_avg_ssse3;
filter8_1dfunction vpx_filter_block1d8_h2_avg_ssse3;
filter8_1dfunction vpx_filter_block1d4_v2_avg_ssse3;
filter8_1dfunction vpx_filter_block1d4_h2_avg_ssse3;
#define vpx_filter_block1d16_v2_avg_avx2 vpx_filter_block1d16_v2_avg_ssse3
#define vpx_filter_block1d16_h2_avg_avx2 vpx_filter_block1d16_h2_avg_ssse3
#define vpx_filter_block1d8_v2_avg_avx2 vpx_filter_block1d8_v2_avg_ssse3
#define vpx_filter_block1d8_h2_avg_avx2 vpx_filter_block1d8_h2_avg_ssse3
#define vpx_filter_block1d4_v2_avg_avx2 vpx_filter_block1d4_v2_avg_ssse3
#define vpx_filter_block1d4_h2_avg_avx2 vpx_filter_block1d4_h2_avg_ssse3
// void vpx_convolve8_horiz_avx2(const uint8_t *src, ptrdiff_t src_stride,
//                                uint8_t *dst, ptrdiff_t dst_stride,
//                                const InterpKernel *filter, int x0_q4,
//                                int32_t x_step_q4, int y0_q4, int y_step_q4,
//                                int w, int h);
// void vpx_convolve8_vert_avx2(const uint8_t *src, ptrdiff_t src_stride,
//                               uint8_t *dst, ptrdiff_t dst_stride,
//                               const InterpKernel *filter, int x0_q4,
//                               int32_t x_step_q4, int y0_q4, int y_step_q4,
//                               int w, int h);
// void vpx_convolve8_avg_horiz_avx2(const uint8_t *src, ptrdiff_t src_stride,
//                                    uint8_t *dst, ptrdiff_t dst_stride,
//                                    const InterpKernel *filter, int x0_q4,
//                                    int32_t x_step_q4, int y0_q4,
//                                    int y_step_q4, int w, int h);
// void vpx_convolve8_avg_vert_avx2(const uint8_t *src, ptrdiff_t src_stride,
//                                   uint8_t *dst, ptrdiff_t dst_stride,
//                                   const InterpKernel *filter, int x0_q4,
//                                   int32_t x_step_q4, int y0_q4,
//                                   int y_step_q4, int w, int h);
FUN_CONV_1D(horiz, x0_q4, x_step_q4, h, src, , avx2);
FUN_CONV_1D(vert, y0_q4, y_step_q4, v, src - src_stride * 3, , avx2);
FUN_CONV_1D(avg_horiz, x0_q4, x_step_q4, h, src, avg_, avx2);
FUN_CONV_1D(avg_vert, y0_q4, y_step_q4, v, src - src_stride * 3, avg_, avx2);

// void vpx_convolve8_avx2(const uint8_t *src, ptrdiff_t src_stride,
//                          uint8_t *dst, ptrdiff_t dst_stride,
//                          const InterpKernel *filter, int x0_q4,
//                          int32_t x_step_q4, int y0_q4, int y_step_q4,
//                          int w, int h);
// void vpx_convolve8_avg_avx2(const uint8_t *src, ptrdiff_t src_stride,
//                              uint8_t *dst, ptrdiff_t dst_stride,
//                              const InterpKernel *filter, int x0_q4,
//                              int32_t x_step_q4, int y0_q4, int y_step_q4,
//                              int w, int h);
FUN_CONV_2D(, avx2);
FUN_CONV_2D(avg_, avx2);
#endif  // HAVE_AX2 && HAVE_SSSE3
