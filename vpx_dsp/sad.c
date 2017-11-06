/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdlib.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"

#include "vpx/vpx_integer.h"
#include "vpx_ports/mem.h"

/* Sum the difference between every corresponding element of the buffers. */
static INLINE unsigned int sad(const uint8_t *a, int a_stride, const uint8_t *b,
                               int b_stride, int width, int height) {
  int y, x;
  unsigned int sad = 0;

  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) sad += abs(a[x] - b[x]);

    a += a_stride;
    b += b_stride;
  }
  return sad;
}

static void od_mc_hadamard_1d(int32_t *diff,
 int n, int stride0, int stride1){
  int i;
  if (n == 4) {
    /*4x4 is the base case as it's our smallest block size.
      Perform the low-level 2x2 butterflies.*/
    int32_t a;
    int32_t b;
    int32_t c;
    int32_t d;
    for (i = 0; i < 4; i++) {
      a = diff[0*stride1] + diff[1*stride1];
      b = diff[0*stride1] - diff[1*stride1];
      c = diff[2*stride1] + diff[3*stride1];
      d = diff[2*stride1] - diff[3*stride1];
      diff[0*stride1] = a + c;
      diff[2*stride1] = a - c;
      diff[1*stride1] = b + d;
      diff[3*stride1] = b - d;
      diff += stride0;
    }
  }
  else {
    /*Recursive case for 8x8, 16x16, etc.
      Subdivide then combine.*/
    int n2;
    int j;
    int32_t *ptr0;
    int32_t *ptr1;
    n2 = n >> 1;
    od_mc_hadamard_1d(diff, n2, stride0, stride1);
    od_mc_hadamard_1d(diff + n2*stride0, n2, stride0, stride1);
    od_mc_hadamard_1d(diff + n2*stride1, n2, stride0, stride1);
    od_mc_hadamard_1d(diff + n2*stride0 + n2*stride1, n2, stride0, stride1);
    ptr0 = diff;
    ptr1 = diff + stride1*n2;
    for (i = 0; i < n; i++){
      for (j = 0; j < n2*stride1; j+=stride1){
        int32_t temp;
        temp = ptr0[j] - ptr1[j];
        ptr0[j] += ptr1[j];
        ptr1[j] = temp;
      }
      ptr0 += stride0;
      ptr1 += stride0;
    }
  }
}

static INLINE unsigned int satd_thing(const uint8_t *a, int a_stride, const uint8_t *b,
                               int b_stride, int width, int height, unsigned int *sad) {
  int y, x, i;
  int32_t work[32*32];
  unsigned int satd;
  int32_t *p;
  p = work;

  for (y = 0; y < height; y+=2) {
    for (x = 0; x < width; x+=2) {
      (*sad) += abs(a[x] - b[x]);
      p[(x >> 1)] = a[x] - b[x];
    }

    p += width;
    a += a_stride*2;
    b += b_stride*2;
  }
  /*Horizontal transform.*/
  od_mc_hadamard_1d(work, width, width, 1);
  /*Vertical transform.*/
  od_mc_hadamard_1d(work, width, 1, width);
  satd = 0;
  for (i = 0; i < width*width; i++) satd += abs(work[i]);
  return satd;
}

static INLINE unsigned int sad_satd(const uint8_t *a, int a_stride, const uint8_t *b,
                               int b_stride, int width, int height) {
  int y, x;
  unsigned int sad = 0;
  unsigned int sadB = 0;
  unsigned int satd = satd_thing(a, a_stride, b, b_stride, width, height, &sadB);
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) sad += abs(a[x] - b[x]);

    a += a_stride;
    b += b_stride;
  }
  if (sadB == 0) {return sad;}
  return (sad*satd)/(sadB);
}

#define sad_satdMxN(m, n)                                                   \
  unsigned int vpx_sad_satd##m##x##n##_c(const uint8_t *src, int src_stride,     \
                                    const uint8_t *ref, int ref_stride) {   \
    return sad_satd(src, src_stride, ref, ref_stride, m, n);                \
  }
#define sad_satdMxNx4D(m, n)                                                    \
  void vpx_sad_satd##m##x##n##x4d_c(const uint8_t *src, int src_stride,         \
                               const uint8_t *const ref_array[],           \
                               int ref_stride, uint32_t *sad_array) {      \
    int i;                                                                 \
    for (i = 0; i < 4; ++i)                                                \
      sad_array[i] =                                                       \
          vpx_sad_satd##m##x##n##_c(src, src_stride, ref_array[i], ref_stride); \
  }

sad_satdMxN(16, 16)
sad_satdMxNx4D(16, 16)

#define sadMxN(m, n)                                                        \
  unsigned int vpx_sad##m##x##n##_c(const uint8_t *src, int src_stride,     \
                                    const uint8_t *ref, int ref_stride) {   \
    return sad(src, src_stride, ref, ref_stride, m, n);                     \
  }                                                                         \
  unsigned int vpx_sad##m##x##n##_avg_c(const uint8_t *src, int src_stride, \
                                        const uint8_t *ref, int ref_stride, \
                                        const uint8_t *second_pred) {       \
    DECLARE_ALIGNED(16, uint8_t, comp_pred[m * n]);                         \
    vpx_comp_avg_pred_c(comp_pred, second_pred, m, n, ref, ref_stride);     \
    return sad(src, src_stride, comp_pred, m, m, n);                        \
  }

// depending on call sites, pass **ref_array to avoid & in subsequent call and
// de-dup with 4D below.
#define sadMxNxK(m, n, k)                                                   \
  void vpx_sad##m##x##n##x##k##_c(const uint8_t *src, int src_stride,       \
                                  const uint8_t *ref_array, int ref_stride, \
                                  uint32_t *sad_array) {                    \
    int i;                                                                  \
    for (i = 0; i < k; ++i)                                                 \
      sad_array[i] =                                                        \
          vpx_sad##m##x##n##_c(src, src_stride, &ref_array[i], ref_stride); \
  }

// This appears to be equivalent to the above when k == 4 and refs is const
#define sadMxNx4D(m, n)                                                    \
  void vpx_sad##m##x##n##x4d_c(const uint8_t *src, int src_stride,         \
                               const uint8_t *const ref_array[],           \
                               int ref_stride, uint32_t *sad_array) {      \
    int i;                                                                 \
    for (i = 0; i < 4; ++i)                                                \
      sad_array[i] =                                                       \
          vpx_sad##m##x##n##_c(src, src_stride, ref_array[i], ref_stride); \
  }

/* clang-format off */
// 64x64
sadMxN(64, 64)
sadMxNx4D(64, 64)

// 64x32
sadMxN(64, 32)
sadMxNx4D(64, 32)

// 32x64
sadMxN(32, 64)
sadMxNx4D(32, 64)

// 32x32
sadMxN(32, 32)
sadMxNx4D(32, 32)

// 32x16
sadMxN(32, 16)
sadMxNx4D(32, 16)

// 16x32
sadMxN(16, 32)
sadMxNx4D(16, 32)

// 16x16
sadMxN(16, 16)
sadMxNxK(16, 16, 3)
sadMxNxK(16, 16, 8)
sadMxNx4D(16, 16)

// 16x8
sadMxN(16, 8)
sadMxNxK(16, 8, 3)
sadMxNxK(16, 8, 8)
sadMxNx4D(16, 8)

// 8x16
sadMxN(8, 16)
sadMxNxK(8, 16, 3)
sadMxNxK(8, 16, 8)
sadMxNx4D(8, 16)

// 8x8
sadMxN(8, 8)
sadMxNxK(8, 8, 3)
sadMxNxK(8, 8, 8)
sadMxNx4D(8, 8)

// 8x4
sadMxN(8, 4)
sadMxNx4D(8, 4)

// 4x8
sadMxN(4, 8)
sadMxNx4D(4, 8)

// 4x4
sadMxN(4, 4)
sadMxNxK(4, 4, 3)
sadMxNxK(4, 4, 8)
sadMxNx4D(4, 4)
/* clang-format on */

#if CONFIG_VP9_HIGHBITDEPTH
        static INLINE
    unsigned int highbd_sad(const uint8_t *a8, int a_stride, const uint8_t *b8,
                            int b_stride, int width, int height) {
  int y, x;
  unsigned int sad = 0;
  const uint16_t *a = CONVERT_TO_SHORTPTR(a8);
  const uint16_t *b = CONVERT_TO_SHORTPTR(b8);
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) sad += abs(a[x] - b[x]);

    a += a_stride;
    b += b_stride;
  }
  return sad;
}

static INLINE unsigned int highbd_sadb(const uint8_t *a8, int a_stride,
                                       const uint16_t *b, int b_stride,
                                       int width, int height) {
  int y, x;
  unsigned int sad = 0;
  const uint16_t *a = CONVERT_TO_SHORTPTR(a8);
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) sad += abs(a[x] - b[x]);

    a += a_stride;
    b += b_stride;
  }
  return sad;
}

#define highbd_sadMxN(m, n)                                                    \
  unsigned int vpx_highbd_sad##m##x##n##_c(const uint8_t *src, int src_stride, \
                                           const uint8_t *ref,                 \
                                           int ref_stride) {                   \
    return highbd_sad(src, src_stride, ref, ref_stride, m, n);                 \
  }                                                                            \
  unsigned int vpx_highbd_sad##m##x##n##_avg_c(                                \
      const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride,  \
      const uint8_t *second_pred) {                                            \
    DECLARE_ALIGNED(16, uint16_t, comp_pred[m * n]);                           \
    vpx_highbd_comp_avg_pred_c(comp_pred, second_pred, m, n, ref, ref_stride); \
    return highbd_sadb(src, src_stride, comp_pred, m, m, n);                   \
  }

#define highbd_sadMxNx4D(m, n)                                               \
  void vpx_highbd_sad##m##x##n##x4d_c(const uint8_t *src, int src_stride,    \
                                      const uint8_t *const ref_array[],      \
                                      int ref_stride, uint32_t *sad_array) { \
    int i;                                                                   \
    for (i = 0; i < 4; ++i) {                                                \
      sad_array[i] = vpx_highbd_sad##m##x##n##_c(src, src_stride,            \
                                                 ref_array[i], ref_stride);  \
    }                                                                        \
  }

/* clang-format off */
// 64x64
highbd_sadMxN(64, 64)
highbd_sadMxNx4D(64, 64)

// 64x32
highbd_sadMxN(64, 32)
highbd_sadMxNx4D(64, 32)

// 32x64
highbd_sadMxN(32, 64)
highbd_sadMxNx4D(32, 64)

// 32x32
highbd_sadMxN(32, 32)
highbd_sadMxNx4D(32, 32)

// 32x16
highbd_sadMxN(32, 16)
highbd_sadMxNx4D(32, 16)

// 16x32
highbd_sadMxN(16, 32)
highbd_sadMxNx4D(16, 32)

// 16x16
highbd_sadMxN(16, 16)
highbd_sadMxNx4D(16, 16)

// 16x8
highbd_sadMxN(16, 8)
highbd_sadMxNx4D(16, 8)

// 8x16
highbd_sadMxN(8, 16)
highbd_sadMxNx4D(8, 16)

// 8x8
highbd_sadMxN(8, 8)
highbd_sadMxNx4D(8, 8)

// 8x4
highbd_sadMxN(8, 4)
highbd_sadMxNx4D(8, 4)

// 4x8
highbd_sadMxN(4, 8)
highbd_sadMxNx4D(4, 8)

// 4x4
highbd_sadMxN(4, 4)
highbd_sadMxNx4D(4, 4)
/* clang-format on */

#endif  // CONFIG_VP9_HIGHBITDEPTH
