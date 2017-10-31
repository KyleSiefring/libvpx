/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <immintrin.h>  // AVX2

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/txfm_common.h"

#define pair256_set_epi16(a, b)                                            \
  _mm256_set_epi16((int16_t)(b), (int16_t)(a), (int16_t)(b), (int16_t)(a), \
                   (int16_t)(b), (int16_t)(a), (int16_t)(b), (int16_t)(a), \
                   (int16_t)(b), (int16_t)(a), (int16_t)(b), (int16_t)(a), \
                   (int16_t)(b), (int16_t)(a), (int16_t)(b), (int16_t)(a))

#if DCT_HIGH_BIT_DEPTH
#define ADD_EPI16 _mm256_adds_epi16
#define SUB_EPI16 _mm256_subs_epi16

#else
#define ADD_EPI16 _mm256_add_epi16
#define SUB_EPI16 _mm256_sub_epi16
#endif

static INLINE __m256i mult_round_shift(const __m256i *pin0, const __m256i *pin1,
                                       const __m256i *pmultiplier,
                                       const __m256i *prounding,
                                       const int shift) {
  const __m256i u0 = _mm256_madd_epi16(*pin0, *pmultiplier);
  const __m256i u1 = _mm256_madd_epi16(*pin1, *pmultiplier);
  const __m256i v0 = _mm256_add_epi32(u0, *prounding);
  const __m256i v1 = _mm256_add_epi32(u1, *prounding);
  const __m256i w0 = _mm256_srai_epi32(v0, shift);
  const __m256i w1 = _mm256_srai_epi32(v1, shift);
  return _mm256_packs_epi32(w0, w1);
}

static INLINE void store_output(const __m256i *poutput, tran_low_t *dst_ptr) {
#if CONFIG_VP9_HIGHBITDEPTH
  const __m256i zero = _mm256_setzero_si256();
  const __m256i sign_bits = _mm256_cmplt_epi16(*poutput, zero);
  __m256i out0 = _mm256_unpacklo_epi16(*poutput, sign_bits);
  __m256i out1 = _mm256_unpackhi_epi16(*poutput, sign_bits);
  _mm256_store_si256((__m256i *)(dst_ptr), out0);
  _mm256_store_si256((__m256i *)(dst_ptr + 8), out1);
#else
  _mm256_store_si256((__m256i *)(dst_ptr), *poutput);
#endif  // CONFIG_VP9_HIGHBITDEPTH
}

static INLINE int check_epi16_overflow_x4(const __m256i *preg0,
                                          const __m256i *preg1,
                                          const __m256i *preg2,
                                          const __m256i *preg3) {
  const __m256i max_overflow = _mm256_set1_epi16(0x7fff);
  const __m256i min_overflow = _mm256_set1_epi16(0x8000);
  __m256i cmp0 = _mm256_or_si256(_mm256_cmpeq_epi16(*preg0, max_overflow),
                              _mm256_cmpeq_epi16(*preg0, min_overflow));
  __m256i cmp1 = _mm256_or_si256(_mm256_cmpeq_epi16(*preg1, max_overflow),
                              _mm256_cmpeq_epi16(*preg1, min_overflow));
  __m256i cmp2 = _mm256_or_si256(_mm256_cmpeq_epi16(*preg2, max_overflow),
                              _mm256_cmpeq_epi16(*preg2, min_overflow));
  __m256i cmp3 = _mm256_or_si256(_mm256_cmpeq_epi16(*preg3, max_overflow),
                              _mm256_cmpeq_epi16(*preg3, min_overflow));
  cmp0 = _mm256_or_si256(_mm256_or_si256(cmp0, cmp1), _mm256_or_si256(cmp2, cmp3));
  return _mm256_movemask_epi8(cmp0);
}

static INLINE int check_epi16_overflow_x8(
    const __m256i *preg0, const __m256i *preg1, const __m256i *preg2,
    const __m256i *preg3, const __m256i *preg4, const __m256i *preg5,
    const __m256i *preg6, const __m256i *preg7) {
  int res0, res1;
  res0 = check_epi16_overflow_x4(preg0, preg1, preg2, preg3);
  res1 = check_epi16_overflow_x4(preg4, preg5, preg6, preg7);
  return res0 + res1;
}

void FDCT16x16_2D_AVX2(const int16_t *input, tran_low_t *output, int stride) {
  // The 2D transform is done with two passes which are actually pretty
  // similar. In the first one, we transform the columns and transpose
  // the results. In the second one, we transform the rows. To achieve that,
  // as the first pass results are transposed, we transpose the columns (that
  // is the transposed rows) and transpose the results (so that it goes back
  // in normal/row positions).
  int pass;
  // Constants
  //    When we use them, in one case, they are all the same. In all others
  //    it's a pair of them that we need to repeat four times. This is done
  //    by constructing the 32 bit constant corresponding to that pair.
  const __m256i k__cospi_p16_p16 = _mm256_set1_epi16(cospi_16_64);
  const __m256i k__cospi_p16_m16 = pair256_set_epi16(cospi_16_64, -cospi_16_64);
  const __m256i k__cospi_p24_p08 = pair256_set_epi16(cospi_24_64, cospi_8_64);
  const __m256i k__cospi_p08_m24 = pair256_set_epi16(cospi_8_64, -cospi_24_64);
  const __m256i k__cospi_m08_p24 = pair256_set_epi16(-cospi_8_64, cospi_24_64);
  const __m256i k__cospi_p28_p04 = pair256_set_epi16(cospi_28_64, cospi_4_64);
  const __m256i k__cospi_m04_p28 = pair256_set_epi16(-cospi_4_64, cospi_28_64);
  const __m256i k__cospi_p12_p20 = pair256_set_epi16(cospi_12_64, cospi_20_64);
  const __m256i k__cospi_m20_p12 = pair256_set_epi16(-cospi_20_64, cospi_12_64);
  const __m256i k__cospi_p30_p02 = pair256_set_epi16(cospi_30_64, cospi_2_64);
  const __m256i k__cospi_p14_p18 = pair256_set_epi16(cospi_14_64, cospi_18_64);
  const __m256i k__cospi_m02_p30 = pair256_set_epi16(-cospi_2_64, cospi_30_64);
  const __m256i k__cospi_m18_p14 = pair256_set_epi16(-cospi_18_64, cospi_14_64);
  const __m256i k__cospi_p22_p10 = pair256_set_epi16(cospi_22_64, cospi_10_64);
  const __m256i k__cospi_p06_p26 = pair256_set_epi16(cospi_6_64, cospi_26_64);
  const __m256i k__cospi_m10_p22 = pair256_set_epi16(-cospi_10_64, cospi_22_64);
  const __m256i k__cospi_m26_p06 = pair256_set_epi16(-cospi_26_64, cospi_6_64);
  const __m256i k__DCT_CONST_ROUNDING = _mm256_set1_epi32(DCT_CONST_ROUNDING);
  const __m256i kOne = _mm256_set1_epi16(1);

  __m256i in00 = _mm256_loadu_si256((const __m256i *)(input + 0 * stride));
  __m256i in01 = _mm256_loadu_si256((const __m256i *)(input + 1 * stride));
  __m256i in02 = _mm256_loadu_si256((const __m256i *)(input + 2 * stride));
  __m256i in03 = _mm256_loadu_si256((const __m256i *)(input + 3 * stride));
  __m256i in04 = _mm256_loadu_si256((const __m256i *)(input + 4 * stride));
  __m256i in05 = _mm256_loadu_si256((const __m256i *)(input + 5 * stride));
  __m256i in06 = _mm256_loadu_si256((const __m256i *)(input + 6 * stride));
  __m256i in07 = _mm256_loadu_si256((const __m256i *)(input + 7 * stride));
  __m256i in08 = _mm256_loadu_si256((const __m256i *)(input + 8 * stride));
  __m256i in09 = _mm256_loadu_si256((const __m256i *)(input + 9 * stride));
  __m256i in10 = _mm256_loadu_si256((const __m256i *)(input + 10 * stride));
  __m256i in11 = _mm256_loadu_si256((const __m256i *)(input + 11 * stride));
  __m256i in12 = _mm256_loadu_si256((const __m256i *)(input + 12 * stride));
  __m256i in13 = _mm256_loadu_si256((const __m256i *)(input + 13 * stride));
  __m256i in14 = _mm256_loadu_si256((const __m256i *)(input + 14 * stride));
  __m256i in15 = _mm256_loadu_si256((const __m256i *)(input + 15 * stride));
  // Do the two transform/transpose passes
  for (pass = 0; pass < 2; ++pass) {
    // We process eight columns (transposed rows in second pass) at a time.
#if DCT_HIGH_BIT_DEPTH
    int overflow;
#endif
    {
      __m256i input0, input1, input2, input3, input4, input5, input6, input7;
      __m256i step1_0, step1_1, step1_2, step1_3;
      __m256i step1_4, step1_5, step1_6, step1_7;
      __m256i step2_1, step2_2, step2_3, step2_4, step2_5, step2_6;
      __m256i step3_0, step3_1, step3_2, step3_3;
      __m256i step3_4, step3_5, step3_6, step3_7;
      __m256i res00, res01, res02, res03, res04, res05, res06, res07;
      __m256i res08, res09, res10, res11, res12, res13, res14, res15;
      // Load and pre-condition input.
      if (0 == pass) {
        // x = x << 2
        in00 = _mm256_slli_epi16(in00, 2);
        in01 = _mm256_slli_epi16(in01, 2);
        in02 = _mm256_slli_epi16(in02, 2);
        in03 = _mm256_slli_epi16(in03, 2);
        in04 = _mm256_slli_epi16(in04, 2);
        in05 = _mm256_slli_epi16(in05, 2);
        in06 = _mm256_slli_epi16(in06, 2);
        in07 = _mm256_slli_epi16(in07, 2);
        in08 = _mm256_slli_epi16(in08, 2);
        in09 = _mm256_slli_epi16(in09, 2);
        in10 = _mm256_slli_epi16(in10, 2);
        in11 = _mm256_slli_epi16(in11, 2);
        in12 = _mm256_slli_epi16(in12, 2);
        in13 = _mm256_slli_epi16(in13, 2);
        in14 = _mm256_slli_epi16(in14, 2);
        in15 = _mm256_slli_epi16(in15, 2);
      } else {
        /*in00 = _mm_load_si128((const __m128i *)(in + 0 * 16));
        in01 = _mm_load_si128((const __m128i *)(in + 1 * 16));
        in02 = _mm_load_si128((const __m128i *)(in + 2 * 16));
        in03 = _mm_load_si128((const __m128i *)(in + 3 * 16));
        in04 = _mm_load_si128((const __m128i *)(in + 4 * 16));
        in05 = _mm_load_si128((const __m128i *)(in + 5 * 16));
        in06 = _mm_load_si128((const __m128i *)(in + 6 * 16));
        in07 = _mm_load_si128((const __m128i *)(in + 7 * 16));
        in08 = _mm_load_si128((const __m128i *)(in + 8 * 16));
        in09 = _mm_load_si128((const __m128i *)(in + 9 * 16));
        in10 = _mm_load_si128((const __m128i *)(in + 10 * 16));
        in11 = _mm_load_si128((const __m128i *)(in + 11 * 16));
        in12 = _mm_load_si128((const __m128i *)(in + 12 * 16));
        in13 = _mm_load_si128((const __m128i *)(in + 13 * 16));
        in14 = _mm_load_si128((const __m128i *)(in + 14 * 16));
        in15 = _mm_load_si128((const __m128i *)(in + 15 * 16));*/
        // x = (x + 1) >> 2
        in00 = _mm256_add_epi16(in00, kOne);
        in01 = _mm256_add_epi16(in01, kOne);
        in02 = _mm256_add_epi16(in02, kOne);
        in03 = _mm256_add_epi16(in03, kOne);
        in04 = _mm256_add_epi16(in04, kOne);
        in05 = _mm256_add_epi16(in05, kOne);
        in06 = _mm256_add_epi16(in06, kOne);
        in07 = _mm256_add_epi16(in07, kOne);
        in08 = _mm256_add_epi16(in08, kOne);
        in09 = _mm256_add_epi16(in09, kOne);
        in10 = _mm256_add_epi16(in10, kOne);
        in11 = _mm256_add_epi16(in11, kOne);
        in12 = _mm256_add_epi16(in12, kOne);
        in13 = _mm256_add_epi16(in13, kOne);
        in14 = _mm256_add_epi16(in14, kOne);
        in15 = _mm256_add_epi16(in15, kOne);
        in00 = _mm256_srai_epi16(in00, 2);
        in01 = _mm256_srai_epi16(in01, 2);
        in02 = _mm256_srai_epi16(in02, 2);
        in03 = _mm256_srai_epi16(in03, 2);
        in04 = _mm256_srai_epi16(in04, 2);
        in05 = _mm256_srai_epi16(in05, 2);
        in06 = _mm256_srai_epi16(in06, 2);
        in07 = _mm256_srai_epi16(in07, 2);
        in08 = _mm256_srai_epi16(in08, 2);
        in09 = _mm256_srai_epi16(in09, 2);
        in10 = _mm256_srai_epi16(in10, 2);
        in11 = _mm256_srai_epi16(in11, 2);
        in12 = _mm256_srai_epi16(in12, 2);
        in13 = _mm256_srai_epi16(in13, 2);
        in14 = _mm256_srai_epi16(in14, 2);
        in15 = _mm256_srai_epi16(in15, 2);
      }
      // Calculate input for the first 8 results.
      {
        input0 = ADD_EPI16(in00, in15);
        input1 = ADD_EPI16(in01, in14);
        input2 = ADD_EPI16(in02, in13);
        input3 = ADD_EPI16(in03, in12);
        input4 = ADD_EPI16(in04, in11);
        input5 = ADD_EPI16(in05, in10);
        input6 = ADD_EPI16(in06, in09);
        input7 = ADD_EPI16(in07, in08);
#if DCT_HIGH_BIT_DEPTH
        overflow = check_epi16_overflow_x8(&input0, &input1, &input2, &input3,
                                           &input4, &input5, &input6, &input7);
        if (overflow) {
          vpx_highbd_fdct16x16_c(input, output, stride);
          return;
        }
#endif  // DCT_HIGH_BIT_DEPTH
      }
      // Calculate input for the next 8 results.
      {
        step1_0 = SUB_EPI16(in07, in08);
        step1_1 = SUB_EPI16(in06, in09);
        step1_2 = SUB_EPI16(in05, in10);
        step1_3 = SUB_EPI16(in04, in11);
        step1_4 = SUB_EPI16(in03, in12);
        step1_5 = SUB_EPI16(in02, in13);
        step1_6 = SUB_EPI16(in01, in14);
        step1_7 = SUB_EPI16(in00, in15);
#if DCT_HIGH_BIT_DEPTH
        overflow =
            check_epi16_overflow_x8(&step1_0, &step1_1, &step1_2, &step1_3,
                                    &step1_4, &step1_5, &step1_6, &step1_7);
        if (overflow) {
          vpx_highbd_fdct16x16_c(input, output, stride);
          return;
        }
#endif  // DCT_HIGH_BIT_DEPTH
      }
      // Work on the first eight values; fdct8(input, even_results);
      {
        // Add/subtract
        const __m256i q0 = ADD_EPI16(input0, input7);
        const __m256i q1 = ADD_EPI16(input1, input6);
        const __m256i q2 = ADD_EPI16(input2, input5);
        const __m256i q3 = ADD_EPI16(input3, input4);
        const __m256i q4 = SUB_EPI16(input3, input4);
        const __m256i q5 = SUB_EPI16(input2, input5);
        const __m256i q6 = SUB_EPI16(input1, input6);
        const __m256i q7 = SUB_EPI16(input0, input7);
#if DCT_HIGH_BIT_DEPTH
        overflow =
            check_epi16_overflow_x8(&q0, &q1, &q2, &q3, &q4, &q5, &q6, &q7);
        if (overflow) {
          vpx_highbd_fdct16x16_c(input, output, stride);
          return;
        }
#endif  // DCT_HIGH_BIT_DEPTH
        // Work on first four results
        {
          // Add/subtract
          const __m256i r0 = ADD_EPI16(q0, q3);
          const __m256i r1 = ADD_EPI16(q1, q2);
          const __m256i r2 = SUB_EPI16(q1, q2);
          const __m256i r3 = SUB_EPI16(q0, q3);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x4(&r0, &r1, &r2, &r3);
          if (overflow) {
            vpx_highbd_fdct16x16_c(input, output, stride);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
          // Interleave to do the multiply by constants which gets us
          // into 32 bits.
          {
            const __m256i t0 = _mm256_unpacklo_epi16(r0, r1);
            const __m256i t1 = _mm256_unpackhi_epi16(r0, r1);
            const __m256i t2 = _mm256_unpacklo_epi16(r2, r3);
            const __m256i t3 = _mm256_unpackhi_epi16(r2, r3);
            res00 = mult_round_shift(&t0, &t1, &k__cospi_p16_p16,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
            res08 = mult_round_shift(&t0, &t1, &k__cospi_p16_m16,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
            res04 = mult_round_shift(&t2, &t3, &k__cospi_p24_p08,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
            res12 = mult_round_shift(&t2, &t3, &k__cospi_m08_p24,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
#if DCT_HIGH_BIT_DEPTH
            overflow = check_epi16_overflow_x4(&res00, &res08, &res04, &res12);
            if (overflow) {
              vpx_highbd_fdct16x16_c(input, output, stride);
              return;
            }
#endif  // DCT_HIGH_BIT_DEPTH
          }
        }
        // Work on next four results
        {
          // Interleave to do the multiply by constants which gets us
          // into 32 bits.
          const __m256i d0 = _mm256_unpacklo_epi16(q6, q5);
          const __m256i d1 = _mm256_unpackhi_epi16(q6, q5);
          const __m256i r0 =
              mult_round_shift(&d0, &d1, &k__cospi_p16_m16,
                               &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          const __m256i r1 =
              mult_round_shift(&d0, &d1, &k__cospi_p16_p16,
                               &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x2(&r0, &r1);
          if (overflow) {
            vpx_highbd_fdct16x16_c(input, output, stride);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
          {
            // Add/subtract
            const __m256i x0 = ADD_EPI16(q4, r0);
            const __m256i x1 = SUB_EPI16(q4, r0);
            const __m256i x2 = SUB_EPI16(q7, r1);
            const __m256i x3 = ADD_EPI16(q7, r1);
#if DCT_HIGH_BIT_DEPTH
            overflow = check_epi16_overflow_x4(&x0, &x1, &x2, &x3);
            if (overflow) {
              vpx_highbd_fdct16x16_c(input, output, stride);
              return;
            }
#endif  // DCT_HIGH_BIT_DEPTH
            // Interleave to do the multiply by constants which gets us
            // into 32 bits.
            {
              const __m256i t0 = _mm256_unpacklo_epi16(x0, x3);
              const __m256i t1 = _mm256_unpackhi_epi16(x0, x3);
              const __m256i t2 = _mm256_unpacklo_epi16(x1, x2);
              const __m256i t3 = _mm256_unpackhi_epi16(x1, x2);
              res02 = mult_round_shift(&t0, &t1, &k__cospi_p28_p04,
                                       &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
              res14 = mult_round_shift(&t0, &t1, &k__cospi_m04_p28,
                                       &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
              res10 = mult_round_shift(&t2, &t3, &k__cospi_p12_p20,
                                       &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
              res06 = mult_round_shift(&t2, &t3, &k__cospi_m20_p12,
                                       &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
#if DCT_HIGH_BIT_DEPTH
              overflow =
                  check_epi16_overflow_x4(&res02, &res14, &res10, &res06);
              if (overflow) {
                vpx_highbd_fdct16x16_c(input, output, stride);
                return;
              }
#endif  // DCT_HIGH_BIT_DEPTH
            }
          }
        }
      }
      // Work on the next eight values; step1 -> odd_results
      {
        // step 2
        {
          const __m256i t0 = _mm256_unpacklo_epi16(step1_5, step1_2);
          const __m256i t1 = _mm256_unpackhi_epi16(step1_5, step1_2);
          const __m256i t2 = _mm256_unpacklo_epi16(step1_4, step1_3);
          const __m256i t3 = _mm256_unpackhi_epi16(step1_4, step1_3);
          step2_2 = mult_round_shift(&t0, &t1, &k__cospi_p16_m16,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          step2_3 = mult_round_shift(&t2, &t3, &k__cospi_p16_m16,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          step2_5 = mult_round_shift(&t0, &t1, &k__cospi_p16_p16,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          step2_4 = mult_round_shift(&t2, &t3, &k__cospi_p16_p16,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
#if DCT_HIGH_BIT_DEPTH
          overflow =
              check_epi16_overflow_x4(&step2_2, &step2_3, &step2_5, &step2_4);
          if (overflow) {
            vpx_highbd_fdct16x16_c(input, output, stride);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        // step 3
        {
          step3_0 = ADD_EPI16(step1_0, step2_3);
          step3_1 = ADD_EPI16(step1_1, step2_2);
          step3_2 = SUB_EPI16(step1_1, step2_2);
          step3_3 = SUB_EPI16(step1_0, step2_3);
          step3_4 = SUB_EPI16(step1_7, step2_4);
          step3_5 = SUB_EPI16(step1_6, step2_5);
          step3_6 = ADD_EPI16(step1_6, step2_5);
          step3_7 = ADD_EPI16(step1_7, step2_4);
#if DCT_HIGH_BIT_DEPTH
          overflow =
              check_epi16_overflow_x8(&step3_0, &step3_1, &step3_2, &step3_3,
                                      &step3_4, &step3_5, &step3_6, &step3_7);
          if (overflow) {
            vpx_highbd_fdct16x16_c(input, output, stride);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        // step 4
        {
          const __m256i t0 = _mm256_unpacklo_epi16(step3_1, step3_6);
          const __m256i t1 = _mm256_unpackhi_epi16(step3_1, step3_6);
          const __m256i t2 = _mm256_unpacklo_epi16(step3_2, step3_5);
          const __m256i t3 = _mm256_unpackhi_epi16(step3_2, step3_5);
          step2_1 = mult_round_shift(&t0, &t1, &k__cospi_m08_p24,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          step2_2 = mult_round_shift(&t2, &t3, &k__cospi_p24_p08,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          step2_6 = mult_round_shift(&t0, &t1, &k__cospi_p24_p08,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          step2_5 = mult_round_shift(&t2, &t3, &k__cospi_p08_m24,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
#if DCT_HIGH_BIT_DEPTH
          overflow =
              check_epi16_overflow_x4(&step2_1, &step2_2, &step2_6, &step2_5);
          if (overflow) {
            vpx_highbd_fdct16x16_c(input, output, stride);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        // step 5
        {
          step1_0 = ADD_EPI16(step3_0, step2_1);
          step1_1 = SUB_EPI16(step3_0, step2_1);
          step1_2 = ADD_EPI16(step3_3, step2_2);
          step1_3 = SUB_EPI16(step3_3, step2_2);
          step1_4 = SUB_EPI16(step3_4, step2_5);
          step1_5 = ADD_EPI16(step3_4, step2_5);
          step1_6 = SUB_EPI16(step3_7, step2_6);
          step1_7 = ADD_EPI16(step3_7, step2_6);
#if DCT_HIGH_BIT_DEPTH
          overflow =
              check_epi16_overflow_x8(&step1_0, &step1_1, &step1_2, &step1_3,
                                      &step1_4, &step1_5, &step1_6, &step1_7);
          if (overflow) {
            vpx_highbd_fdct16x16_c(input, output, stride);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        // step 6
        {
          const __m256i t0 = _mm256_unpacklo_epi16(step1_0, step1_7);
          const __m256i t1 = _mm256_unpackhi_epi16(step1_0, step1_7);
          const __m256i t2 = _mm256_unpacklo_epi16(step1_1, step1_6);
          const __m256i t3 = _mm256_unpackhi_epi16(step1_1, step1_6);
          res01 = mult_round_shift(&t0, &t1, &k__cospi_p30_p02,
                                   &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          res09 = mult_round_shift(&t2, &t3, &k__cospi_p14_p18,
                                   &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          res15 = mult_round_shift(&t0, &t1, &k__cospi_m02_p30,
                                   &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          res07 = mult_round_shift(&t2, &t3, &k__cospi_m18_p14,
                                   &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x4(&res01, &res09, &res15, &res07);
          if (overflow) {
            vpx_highbd_fdct16x16_c(input, output, stride);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          const __m256i t0 = _mm256_unpacklo_epi16(step1_2, step1_5);
          const __m256i t1 = _mm256_unpackhi_epi16(step1_2, step1_5);
          const __m256i t2 = _mm256_unpacklo_epi16(step1_3, step1_4);
          const __m256i t3 = _mm256_unpackhi_epi16(step1_3, step1_4);
          res05 = mult_round_shift(&t0, &t1, &k__cospi_p22_p10,
                                   &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          res13 = mult_round_shift(&t2, &t3, &k__cospi_p06_p26,
                                   &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          res11 = mult_round_shift(&t0, &t1, &k__cospi_m10_p22,
                                   &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          res03 = mult_round_shift(&t2, &t3, &k__cospi_m26_p06,
                                   &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x4(&res05, &res13, &res11, &res03);
          if (overflow) {
            vpx_highbd_fdct16x16_c(input, output, stride);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
      }
      // Transpose the results, do it as two 8x8 transposes.
      {
        // 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F
        // 10 11 12 13 14 15 16 17
        // 20 21 22 23 24 25 26 27
        // 30 31 32 33 34 35 36 37
        // 40 41 42 43 44 45 46 47
        // 50 51 52 53 54 55 56 57
        // 60 61 62 63 64 65 66 67
        // 70 71 72 73 74 75 76 77
        // 80 81 82 83 84 85 86 87
        const __m256i tr0_00 = _mm256_unpacklo_epi16(res00, res01);
        const __m256i tr0_01 = _mm256_unpacklo_epi16(res02, res03);
        const __m256i tr0_02 = _mm256_unpackhi_epi16(res00, res01);
        const __m256i tr0_03 = _mm256_unpackhi_epi16(res02, res03);
        const __m256i tr0_04 = _mm256_unpacklo_epi16(res04, res05);
        const __m256i tr0_05 = _mm256_unpacklo_epi16(res06, res07);
        const __m256i tr0_06 = _mm256_unpackhi_epi16(res04, res05);
        const __m256i tr0_07 = _mm256_unpackhi_epi16(res06, res07);
        const __m256i tr0_08 = _mm256_unpacklo_epi16(res08, res09);
        const __m256i tr0_09 = _mm256_unpacklo_epi16(res10, res11);
        const __m256i tr0_10 = _mm256_unpackhi_epi16(res08, res09);
        const __m256i tr0_11 = _mm256_unpackhi_epi16(res10, res11);
        const __m256i tr0_12 = _mm256_unpacklo_epi16(res12, res13);
        const __m256i tr0_13 = _mm256_unpacklo_epi16(res14, res15);
        const __m256i tr0_14 = _mm256_unpackhi_epi16(res12, res13);
        const __m256i tr0_15 = _mm256_unpackhi_epi16(res14, res15);
        // 00 10 01 11 02 12 03 13
        // 20 30 21 31 22 32 23 33
        // 04 14 05 15 06 16 07 17
        // 24 34 25 35 26 36 27 37
        // 40 50 41 51 42 52 43 53
        // 60 70 61 71 62 72 63 73
        // 54 54 55 55 56 56 57 57
        // 64 74 65 75 66 76 67 77
        const __m256i tr1_00 = _mm256_unpacklo_epi32(tr0_00, tr0_01);
        const __m256i tr1_01 = _mm256_unpacklo_epi32(tr0_02, tr0_03);
        const __m256i tr1_02 = _mm256_unpackhi_epi32(tr0_00, tr0_01);
        const __m256i tr1_03 = _mm256_unpackhi_epi32(tr0_02, tr0_03);
        const __m256i tr1_04 = _mm256_unpacklo_epi32(tr0_04, tr0_05);
        const __m256i tr1_05 = _mm256_unpacklo_epi32(tr0_06, tr0_07);
        const __m256i tr1_06 = _mm256_unpackhi_epi32(tr0_04, tr0_05);
        const __m256i tr1_07 = _mm256_unpackhi_epi32(tr0_06, tr0_07);
        const __m256i tr1_08 = _mm256_unpacklo_epi32(tr0_08, tr0_09);
        const __m256i tr1_09 = _mm256_unpacklo_epi32(tr0_10, tr0_11);
        const __m256i tr1_10 = _mm256_unpackhi_epi32(tr0_08, tr0_09);
        const __m256i tr1_11 = _mm256_unpackhi_epi32(tr0_10, tr0_11);
        const __m256i tr1_12 = _mm256_unpacklo_epi32(tr0_12, tr0_13);
        const __m256i tr1_13 = _mm256_unpacklo_epi32(tr0_14, tr0_15);
        const __m256i tr1_14 = _mm256_unpackhi_epi32(tr0_12, tr0_13);
        const __m256i tr1_15 = _mm256_unpackhi_epi32(tr0_14, tr0_15);
        // 00 10 20 30 01 11 21 31
        // 40 50 60 70 41 51 61 71
        // 02 12 22 32 03 13 23 33
        // 42 52 62 72 43 53 63 73
        // 04 14 24 34 05 15 21 36
        // 44 54 64 74 45 55 61 76
        // 06 16 26 36 07 17 27 37
        // 46 56 66 76 47 57 67 77
        const __m256i tr2_00 = _mm256_unpacklo_epi64(tr1_00, tr1_04);
        const __m256i tr2_01 = _mm256_unpackhi_epi64(tr1_00, tr1_04);
        const __m256i tr2_02 = _mm256_unpacklo_epi64(tr1_02, tr1_06);
        const __m256i tr2_03 = _mm256_unpackhi_epi64(tr1_02, tr1_06);
        const __m256i tr2_04 = _mm256_unpacklo_epi64(tr1_01, tr1_05);
        const __m256i tr2_05 = _mm256_unpackhi_epi64(tr1_01, tr1_05);
        const __m256i tr2_06 = _mm256_unpacklo_epi64(tr1_03, tr1_07);
        const __m256i tr2_07 = _mm256_unpackhi_epi64(tr1_03, tr1_07);

        const __m256i tr2_08 = _mm256_unpacklo_epi64(tr1_08, tr1_12);
        const __m256i tr2_09 = _mm256_unpackhi_epi64(tr1_08, tr1_12);
        const __m256i tr2_10 = _mm256_unpacklo_epi64(tr1_10, tr1_14);
        const __m256i tr2_11 = _mm256_unpackhi_epi64(tr1_10, tr1_14);
        const __m256i tr2_12 = _mm256_unpacklo_epi64(tr1_09, tr1_13);
        const __m256i tr2_13 = _mm256_unpackhi_epi64(tr1_09, tr1_13);
        const __m256i tr2_14 = _mm256_unpacklo_epi64(tr1_11, tr1_15);
        const __m256i tr2_15 = _mm256_unpackhi_epi64(tr1_11, tr1_15);
        // 00 10 20 30 40 50 60 70 08
        // 01 11 21 31 41 51 61 71 09
        // 02 12 22 32 42 52 62 72 0A
        // 03 13 23 33 43 53 63 73 0B
        // 04 14 24 34 44 54 64 74 0C
        // 05 15 25 35 45 55 65 75 0D
        // 06 16 26 36 46 56 66 76 0E
        // 07 17 27 37 47 57 67 77 0F
        // 80
        // 81
        in00 = _mm256_permute2x128_si256(tr2_00, tr2_08, 0 | (2 << 4));
        in01 = _mm256_permute2x128_si256(tr2_01, tr2_09, 0 | (2 << 4));
        in02 = _mm256_permute2x128_si256(tr2_02, tr2_10, 0 | (2 << 4));
        in03 = _mm256_permute2x128_si256(tr2_03, tr2_11, 0 | (2 << 4));
        in04 = _mm256_permute2x128_si256(tr2_04, tr2_12, 0 | (2 << 4));
        in05 = _mm256_permute2x128_si256(tr2_05, tr2_13, 0 | (2 << 4));
        in06 = _mm256_permute2x128_si256(tr2_06, tr2_14, 0 | (2 << 4));
        in07 = _mm256_permute2x128_si256(tr2_07, tr2_15, 0 | (2 << 4));

        in08 = _mm256_permute2x128_si256(tr2_00, tr2_08, 1 | (3 << 4));
        in09 = _mm256_permute2x128_si256(tr2_01, tr2_09, 1 | (3 << 4));
        in10 = _mm256_permute2x128_si256(tr2_02, tr2_10, 1 | (3 << 4));
        in11 = _mm256_permute2x128_si256(tr2_03, tr2_11, 1 | (3 << 4));
        in12 = _mm256_permute2x128_si256(tr2_04, tr2_12, 1 | (3 << 4));
        in13 = _mm256_permute2x128_si256(tr2_05, tr2_13, 1 | (3 << 4));
        in14 = _mm256_permute2x128_si256(tr2_06, tr2_14, 1 | (3 << 4));
        in15 = _mm256_permute2x128_si256(tr2_07, tr2_15, 1 | (3 << 4));

      }
      /*transpose_and_output8x8(&res00, &res01, &res02, &res03, &res04, &res05,
                              &res06, &res07, pass, out0, out1);
      transpose_and_output8x8(&res08, &res09, &res10, &res11, &res12, &res13,
                              &res14, &res15, pass, out0 + 8, out1 + 8);*/
    }
  }
  store_output(&in00, (output + 0 * 16));
  store_output(&in01, (output + 1 * 16));
  store_output(&in02, (output + 2 * 16));
  store_output(&in03, (output + 3 * 16));
  store_output(&in04, (output + 4 * 16));
  store_output(&in05, (output + 5 * 16));
  store_output(&in06, (output + 6 * 16));
  store_output(&in07, (output + 7 * 16));
  store_output(&in08, (output + 8 * 16));
  store_output(&in09, (output + 9 * 16));
  store_output(&in10, (output + 10 * 16));
  store_output(&in11, (output + 11 * 16));
  store_output(&in12, (output + 12 * 16));
  store_output(&in13, (output + 13 * 16));
  store_output(&in14, (output + 14 * 16));
  store_output(&in15, (output + 15 * 16));
}
