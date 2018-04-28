/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <immintrin.h>  // AVX2

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/daala_tx.h"

static INLINE __m256i od_mm256_unbiased_rshift1_epi16(__m256i a) {
  return _mm256_srai_epi16(_mm256_add_epi16(_mm256_srli_epi16(a, 15), a), 1);
}

static INLINE __m256i od_mm256_add_avg_epi16(__m256i a, __m256i b) {
  __m256i sign_mask;
  sign_mask = _mm256_set1_epi16(0x7FFF + TX_AVG_BIAS);
  return _mm256_xor_si256(_mm256_avg_epu16(_mm256_xor_si256(a, sign_mask),
                                           _mm256_xor_si256(b, sign_mask)),
                          sign_mask);
}

static INLINE __m256i od_mm256_sub_avg_epi16(__m256i a, __m256i b) {
#if TX_AVG_BIAS
  __m256i sign_bit;
  sign_bit = _mm256_set1_epi16(0x8000);
  return _mm256_xor_si256(_mm256_avg_epu16(_mm256_xor_si256(a, sign_bit),
                                           _mm256_sub_epi16(sign_bit, b)),
                          sign_bit);
#else
  __m256i sign_bit;
  __m256i sign_mask;
  sign_bit = _mm256_set1_epi16(0x8000);
  sign_mask = _mm256_set1_epi16(0x7FFF);
  return _mm256_xor_si256(_mm256_avg_epu16(_mm256_xor_si256(a, sign_bit),
                                           _mm256_xor_si256(b, sign_mask)),
                          sign_bit);
#endif
}

static INLINE void od_mm256_swap_si256(__m256i *q0, __m256i *q1) {
  __m256i t;
  t = *q0;
  *q0 = *q1;
  *q1 = t;
}

static INLINE __m256i od_mm256_mulhrs_epi16(__m256i a, int16_t b) {
  return _mm256_mulhrs_epi16(a, _mm256_set1_epi16(b));
}

static INLINE __m256i od_mm256_mul_epi16(__m256i a, int32_t b, int r) {
  int32_t b_q15;
  if (b > 32767/2) {
    b = (b+1)/2;
    r = r-1;
  }
  b_q15 = b << (15 - r);
  /* b and r are in all cases compile-time constants, so these branches
     disappear when this function gets inlined. */
  if (b_q15 > 32767) {
    return _mm256_add_epi16(a,
                            od_mm256_mulhrs_epi16(a, (int16_t)(b_q15 - 32768)));
  } else if (b_q15 < -32767) {
    return _mm256_sub_epi16(od_mm256_mulhrs_epi16(a, (int16_t)(32768 + b_q15)),
                            a);
  } else {
    return od_mm256_mulhrs_epi16(a, b_q15);
  }
}

#undef OD_KERNEL
#undef OD_COEFF
#undef OD_ADD
#undef OD_SUB
#undef OD_RSHIFT1
#undef OD_ADD_AVG
#undef OD_SUB_AVG
#undef OD_MUL
#undef OD_SWAP

/* Define 16-wide 16-bit AVX2 kernels. */
#define OD_KERNEL kernel16_epi16
#define OD_COEFF __m256i
#define OD_ADD _mm256_add_epi16
#define OD_SUB _mm256_sub_epi16
#define OD_RSHIFT1 od_mm256_unbiased_rshift1_epi16
#define OD_ADD_AVG od_mm256_add_avg_epi16
#define OD_SUB_AVG od_mm256_sub_avg_epi16
#define OD_MUL od_mm256_mul_epi16
#define OD_SWAP od_mm256_swap_si256

#include "vpx_dsp/daala_tx_kernels.h"

static __m256i load_pass1(const int16_t *ptr) {
  __m256i ret;
  ret = _mm256_loadu_si256((const __m256i *)ptr);
  // TODO: shift isn't constant
  return _mm256_slli_epi16(ret, 4);
}

static __m256i load_pass2(const int16_t *ptr) {
  __m256i ret;
  ret = _mm256_loadu_si256((const __m256i *)ptr);
  return _mm256_srai_epi16(ret, 2);
}

static void rd_fdct32_avx2(__m256i y[32], const int16_t *x, int xstride, int pass) {
  __m256i t0;
  __m256i t1;
  __m256i t2;
  __m256i t3;
  __m256i t4;
  __m256i t5;
  __m256i t6;
  __m256i t7;
  __m256i t8;
  __m256i t9;
  __m256i ta;
  __m256i tb;
  __m256i tc;
  __m256i td;
  __m256i te;
  __m256i tf;
  __m256i tg;
  __m256i th;
  __m256i ti;
  __m256i tj;
  __m256i tk;
  __m256i tl;
  __m256i tm;
  __m256i tn;
  __m256i to;
  __m256i tp;
  __m256i tq;
  __m256i tr;
  __m256i ts;
  __m256i tt;
  __m256i tu;
  __m256i tv;
  if (pass == 0) {
    t0 = load_pass1(x + 0*xstride);
    tg = load_pass1(x + 1*xstride);
    t8 = load_pass1(x + 2*xstride);
    to = load_pass1(x + 3*xstride);
    t4 = load_pass1(x + 4*xstride);
    tk = load_pass1(x + 5*xstride);
    tc = load_pass1(x + 6*xstride);
    ts = load_pass1(x + 7*xstride);
    t2 = load_pass1(x + 8*xstride);
    ti = load_pass1(x + 9*xstride);
    ta = load_pass1(x + 10*xstride);
    tq = load_pass1(x + 11*xstride);
    t6 = load_pass1(x + 12*xstride);
    tm = load_pass1(x + 13*xstride);
    te = load_pass1(x + 14*xstride);
    tu = load_pass1(x + 15*xstride);
    t1 = load_pass1(x + 16*xstride);
    th = load_pass1(x + 17*xstride);
    t9 = load_pass1(x + 18*xstride);
    tp = load_pass1(x + 19*xstride);
    t5 = load_pass1(x + 20*xstride);
    tl = load_pass1(x + 21*xstride);
    td = load_pass1(x + 22*xstride);
    tt = load_pass1(x + 23*xstride);
    t3 = load_pass1(x + 24*xstride);
    tj = load_pass1(x + 25*xstride);
    tb = load_pass1(x + 26*xstride);
    tr = load_pass1(x + 27*xstride);
    t7 = load_pass1(x + 28*xstride);
    tn = load_pass1(x + 29*xstride);
    tf = load_pass1(x + 30*xstride);
    tv = load_pass1(x + 31*xstride);
  }
  else {
    t0 = load_pass2(x + 0*xstride);
    tg = load_pass2(x + 1*xstride);
    t8 = load_pass2(x + 2*xstride);
    to = load_pass2(x + 3*xstride);
    t4 = load_pass2(x + 4*xstride);
    tk = load_pass2(x + 5*xstride);
    tc = load_pass2(x + 6*xstride);
    ts = load_pass2(x + 7*xstride);
    t2 = load_pass2(x + 8*xstride);
    ti = load_pass2(x + 9*xstride);
    ta = load_pass2(x + 10*xstride);
    tq = load_pass2(x + 11*xstride);
    t6 = load_pass2(x + 12*xstride);
    tm = load_pass2(x + 13*xstride);
    te = load_pass2(x + 14*xstride);
    tu = load_pass2(x + 15*xstride);
    t1 = load_pass2(x + 16*xstride);
    th = load_pass2(x + 17*xstride);
    t9 = load_pass2(x + 18*xstride);
    tp = load_pass2(x + 19*xstride);
    t5 = load_pass2(x + 20*xstride);
    tl = load_pass2(x + 21*xstride);
    td = load_pass2(x + 22*xstride);
    tt = load_pass2(x + 23*xstride);
    t3 = load_pass2(x + 24*xstride);
    tj = load_pass2(x + 25*xstride);
    tb = load_pass2(x + 26*xstride);
    tr = load_pass2(x + 27*xstride);
    t7 = load_pass2(x + 28*xstride);
    tn = load_pass2(x + 29*xstride);
    tf = load_pass2(x + 30*xstride);
    tv = load_pass2(x + 31*xstride);
  }
  od_fdct_32_kernel16_epi16(
    &t0, &tg, &t8, &to, &t4, &tk, &tc, &ts, &t2, &ti, &ta, &tq, &t6, &tm, &te,
    &tu, &t1, &th, &t9, &tp, &t5, &tl, &td, &tt, &t3, &tj, &tb, &tr, &t7, &tn,
    &tf, &tv);
  y[0] = t0;
  y[1] = t1;
  y[2] = t2;
  y[3] = t3;
  y[4] = t4;
  y[5] = t5;
  y[6] = t6;
  y[7] = t7;
  y[8] = t8;
  y[9] = t9;
  y[10] = ta;
  y[11] = tb;
  y[12] = tc;
  y[13] = td;
  y[14] = te;
  y[15] = tf;
  y[16] = tg;
  y[17] = th;
  y[18] = ti;
  y[19] = tj;
  y[20] = tk;
  y[21] = tl;
  y[22] = tm;
  y[23] = tn;
  y[24] = to;
  y[25] = tp;
  y[26] = tq;
  y[27] = tr;
  y[28] = ts;
  y[29] = tt;
  y[30] = tu;
  y[31] = tv;
}

void FDCT32x32_2D_AVX2(const int16_t *input, int16_t *output_org, int stride) {
  // We need an intermediate buffer between passes.
  DECLARE_ALIGNED(32, int16_t, intermediate[32 * 32]);
  // Do the two transform/transpose passes
  int pass;
  for (pass = 0; pass < 2; ++pass) {
    // We process sixteen columns (transposed rows in second pass) at a time.
    int column_start;
    for (column_start = 0; column_start < 32; column_start += 16) {
      __m256i out[32];
      const int16_t *in = pass == 0 ? &input[column_start] : &intermediate[column_start];
      rd_fdct32_avx2(out, in, pass == 0 ? stride : 32, pass);
      // Transpose the results, do it as four 8x8 transposes.
      {
        int transpose_block;
        int16_t *output_currStep, *output_nextStep;
        if (0 == pass) {
          output_currStep = &intermediate[column_start * 32];
          output_nextStep = &intermediate[(column_start + 8) * 32];
        } else {
          output_currStep = &output_org[column_start * 32];
          output_nextStep = &output_org[(column_start + 8) * 32];
        }
        for (transpose_block = 0; transpose_block < 4; ++transpose_block) {
          __m256i *this_out = &out[8 * transpose_block];
          // 00  01  02  03  04  05  06  07  08  09  10  11  12  13  14  15
          // 20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
          // 40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55
          // 60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75
          // 80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95
          // 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115
          // 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135
          // 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155
          const __m256i tr0_0 = _mm256_unpacklo_epi16(this_out[0], this_out[1]);
          const __m256i tr0_1 = _mm256_unpacklo_epi16(this_out[2], this_out[3]);
          const __m256i tr0_2 = _mm256_unpackhi_epi16(this_out[0], this_out[1]);
          const __m256i tr0_3 = _mm256_unpackhi_epi16(this_out[2], this_out[3]);
          const __m256i tr0_4 = _mm256_unpacklo_epi16(this_out[4], this_out[5]);
          const __m256i tr0_5 = _mm256_unpacklo_epi16(this_out[6], this_out[7]);
          const __m256i tr0_6 = _mm256_unpackhi_epi16(this_out[4], this_out[5]);
          const __m256i tr0_7 = _mm256_unpackhi_epi16(this_out[6], this_out[7]);
          // 00  20  01  21  02  22  03  23  08  28  09  29  10  30  11  31
          // 40  60  41  61  42  62  43  63  48  68  49  69  50  70  51  71
          // 04  24  05  25  06  26  07  27  12  32  13  33  14  34  15  35
          // 44  64  45  65  46  66  47  67  52  72  53  73  54  74  55  75
          // 80  100 81  101 82  102 83  103 88  108 89  109 90  110 91  101
          // 120 140 121 141 122 142 123 143 128 148 129 149 130 150 131 151
          // 84  104 85  105 86  106 87  107 92  112 93  113 94  114 95  115
          // 124 144 125 145 126 146 127 147 132 152 133 153 134 154 135 155

          const __m256i tr1_0 = _mm256_unpacklo_epi32(tr0_0, tr0_1);
          const __m256i tr1_1 = _mm256_unpacklo_epi32(tr0_2, tr0_3);
          const __m256i tr1_2 = _mm256_unpackhi_epi32(tr0_0, tr0_1);
          const __m256i tr1_3 = _mm256_unpackhi_epi32(tr0_2, tr0_3);
          const __m256i tr1_4 = _mm256_unpacklo_epi32(tr0_4, tr0_5);
          const __m256i tr1_5 = _mm256_unpacklo_epi32(tr0_6, tr0_7);
          const __m256i tr1_6 = _mm256_unpackhi_epi32(tr0_4, tr0_5);
          const __m256i tr1_7 = _mm256_unpackhi_epi32(tr0_6, tr0_7);
          // 00 20  40  60  01 21  41  61  08 28  48  68  09 29  49  69
          // 04 24  44  64  05 25  45  65  12 32  52  72  13 33  53  73
          // 02 22  42  62  03 23  43  63  10 30  50  70  11 31  51  71
          // 06 26  46  66  07 27  47  67  14 34  54  74  15 35  55  75
          // 80 100 120 140 81 101 121 141 88 108 128 148 89 109 129 149
          // 84 104 124 144 85 105 125 145 92 112 132 152 93 113 133 153
          // 82 102 122 142 83 103 123 143 90 110 130 150 91 101 131 151
          // 86 106 126 146 87 107 127 147 94 114 134 154 95 115 135 155
          __m256i tr2_0 = _mm256_unpacklo_epi64(tr1_0, tr1_4);
          __m256i tr2_1 = _mm256_unpackhi_epi64(tr1_0, tr1_4);
          __m256i tr2_2 = _mm256_unpacklo_epi64(tr1_2, tr1_6);
          __m256i tr2_3 = _mm256_unpackhi_epi64(tr1_2, tr1_6);
          __m256i tr2_4 = _mm256_unpacklo_epi64(tr1_1, tr1_5);
          __m256i tr2_5 = _mm256_unpackhi_epi64(tr1_1, tr1_5);
          __m256i tr2_6 = _mm256_unpacklo_epi64(tr1_3, tr1_7);
          __m256i tr2_7 = _mm256_unpackhi_epi64(tr1_3, tr1_7);
          // 00 20 40 60 80 100 120 140 08 28 48 68 88 108 128 148
          // 01 21 41 61 81 101 121 141 09 29 49 69 89 109 129 149
          // 02 22 42 62 82 102 122 142 10 30 50 70 90 110 130 150
          // 03 23 43 63 83 103 123 143 11 31 51 71 91 101 131 151
          // 04 24 44 64 84 104 124 144 12 32 52 72 92 112 132 152
          // 05 25 45 65 85 105 125 145 13 33 53 73 93 113 133 153
          // 06 26 46 66 86 106 126 146 14 34 54 74 94 114 134 154
          // 07 27 47 67 87 107 127 147 15 35 55 75 95 115 135 155
          // Note: even though all these stores are aligned, using the aligned
          //       intrinsic make the code slightly slower.
          _mm_storeu_si128((__m128i *)(output_currStep + 0 * 32),
                           _mm256_castsi256_si128(tr2_0));
          _mm_storeu_si128((__m128i *)(output_currStep + 1 * 32),
                           _mm256_castsi256_si128(tr2_1));
          _mm_storeu_si128((__m128i *)(output_currStep + 2 * 32),
                           _mm256_castsi256_si128(tr2_2));
          _mm_storeu_si128((__m128i *)(output_currStep + 3 * 32),
                           _mm256_castsi256_si128(tr2_3));
          _mm_storeu_si128((__m128i *)(output_currStep + 4 * 32),
                           _mm256_castsi256_si128(tr2_4));
          _mm_storeu_si128((__m128i *)(output_currStep + 5 * 32),
                           _mm256_castsi256_si128(tr2_5));
          _mm_storeu_si128((__m128i *)(output_currStep + 6 * 32),
                           _mm256_castsi256_si128(tr2_6));
          _mm_storeu_si128((__m128i *)(output_currStep + 7 * 32),
                           _mm256_castsi256_si128(tr2_7));

          _mm_storeu_si128((__m128i *)(output_nextStep + 0 * 32),
                           _mm256_extractf128_si256(tr2_0, 1));
          _mm_storeu_si128((__m128i *)(output_nextStep + 1 * 32),
                           _mm256_extractf128_si256(tr2_1, 1));
          _mm_storeu_si128((__m128i *)(output_nextStep + 2 * 32),
                           _mm256_extractf128_si256(tr2_2, 1));
          _mm_storeu_si128((__m128i *)(output_nextStep + 3 * 32),
                           _mm256_extractf128_si256(tr2_3, 1));
          _mm_storeu_si128((__m128i *)(output_nextStep + 4 * 32),
                           _mm256_extractf128_si256(tr2_4, 1));
          _mm_storeu_si128((__m128i *)(output_nextStep + 5 * 32),
                           _mm256_extractf128_si256(tr2_5, 1));
          _mm_storeu_si128((__m128i *)(output_nextStep + 6 * 32),
                           _mm256_extractf128_si256(tr2_6, 1));
          _mm_storeu_si128((__m128i *)(output_nextStep + 7 * 32),
                           _mm256_extractf128_si256(tr2_7, 1));
          // Process next 8x8
          output_currStep += 8;
          output_nextStep += 8;
        }
      }
    }
  }
}
