/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <immintrin.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/x86/bitdepth_conversion_avx2.h"

static INLINE void load_b_values(const int16_t *zbin_ptr, __m256i *zbin,
                                 const int16_t *round_ptr, __m256i *round,
                                 const int16_t *quant_ptr, __m256i *quant,
                                 const int16_t *dequant_ptr, __m256i *dequant,
                                 const int16_t *shift_ptr, __m256i *shift) {
  *zbin = _mm256_load_si256((const __m256i *)zbin_ptr);
  *round = _mm256_load_si256((const __m256i *)round_ptr);
  *quant = _mm256_load_si256((const __m256i *)quant_ptr);
  *zbin = _mm256_sub_epi16(*zbin, _mm256_set1_epi16(1));
  *dequant = _mm256_load_si256((const __m256i *)dequant_ptr);
  *shift = _mm256_load_si256((const __m256i *)shift_ptr);
}

static INLINE void calculate_qcoeff(__m256i *coeff, const __m256i round,
                                    const __m256i quant, const __m256i shift) {
  __m256i tmp, qcoeff;
  qcoeff = _mm256_adds_epi16(*coeff, round);
  tmp = _mm256_mulhi_epi16(qcoeff, quant);
  qcoeff = _mm256_add_epi16(tmp, qcoeff);
  *coeff = _mm256_mulhi_epi16(qcoeff, shift);
}

static INLINE __m256i calculate_dqcoeff(__m256i qcoeff, __m256i dequant) {
  return _mm256_mullo_epi16(qcoeff, dequant);
}

static INLINE __m256i scan_for_eob_single(__m256i *coeff,
                                   const __m256i zbin_mask,
                                   const int16_t *scan_ptr, const int index,
                                   const __m256i zero) {
  const __m256i zero_coeff = _mm256_cmpeq_epi16(*coeff, zero);
  __m256i scan = _mm256_load_si256((const __m256i *)(scan_ptr + index));
  // Add one to convert from indices to counts
  scan = _mm256_sub_epi16(scan, zbin_mask);
  return _mm256_andnot_si256(zero_coeff, scan);
}

// Scan 16 values for eob reference in scan_ptr. Use masks (-1) from comparing
// to zbin to add 1 to the index in 'scan'.
static INLINE __m256i scan_for_eob(__m256i *coeff0, __m256i *coeff1,
                                   const __m256i zbin_mask0,
                                   const __m256i zbin_mask1,
                                   const int16_t *scan_ptr, const int index,
                                   const __m256i zero) {
  const __m256i zero_coeff0 = _mm256_cmpeq_epi16(*coeff0, zero);
  const __m256i zero_coeff1 = _mm256_cmpeq_epi16(*coeff1, zero);
  __m256i scan0 = _mm256_load_si256((const __m256i *)(scan_ptr + index));
  __m256i scan1 = _mm256_load_si256((const __m256i *)(scan_ptr + index + 16));
  __m256i eob0, eob1;
  // Add one to convert from indices to counts
  scan0 = _mm256_sub_epi16(scan0, zbin_mask0);
  scan1 = _mm256_sub_epi16(scan1, zbin_mask1);
  eob0 = _mm256_andnot_si256(zero_coeff0, scan0);
  eob1 = _mm256_andnot_si256(zero_coeff1, scan1);
  return _mm256_max_epi16(eob0, eob1);
}

static INLINE int16_t accumulate_eob(__m256i eob256) {
  __m128i eob, eob_shuffled;
  eob = _mm_max_epi16(_mm256_castsi256_si128(eob256), _mm256_extractf128_si256(eob256, 1));
  eob_shuffled = _mm_shuffle_epi32(eob, 0xe);
  eob = _mm_max_epi16(eob, eob_shuffled);
  eob_shuffled = _mm_shufflelo_epi16(eob, 0xe);
  eob = _mm_max_epi16(eob, eob_shuffled);
  eob_shuffled = _mm_shufflelo_epi16(eob, 0x1);
  eob = _mm_max_epi16(eob, eob_shuffled);
  return _mm_extract_epi16(eob, 1);
}

void vpx_quantize_b_avx2(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                        int skip_block, const int16_t *zbin_ptr,
                        const int16_t *round_ptr, const int16_t *quant_ptr,
                        const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
                        tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr,
                        uint16_t *eob_ptr, const int16_t *scan_ptr,
                        const int16_t *iscan_ptr) {
  const __m256i zero = _mm256_setzero_si256();
  int index;

  __m256i zbin, round, quant, dequant, shift;
  __m256i coeff0, coeff1;
  __m256i qcoeff0, qcoeff1;
  __m256i cmp_mask0, cmp_mask1;
  __m256i all_zero;
  __m256i eob = zero, eob0;

  (void)scan_ptr;
  (void)skip_block;
  assert(!skip_block);

  *eob_ptr = 0;

  load_b_values(zbin_ptr, &zbin, round_ptr, &round, quant_ptr, &quant,
                dequant_ptr, &dequant, quant_shift_ptr, &shift);

  if (n_coeffs == 16) {
    // Do DC and first 15 AC.
    coeff0 = load_tran_low(coeff_ptr);

    qcoeff0 = _mm256_abs_epi16(coeff0);

    cmp_mask0 = _mm256_cmpgt_epi16(qcoeff0, zbin);

    if (_mm256_testz_si256(cmp_mask0, cmp_mask0)) {
      _mm256_store_si256((__m256i *)(qcoeff_ptr), zero);
      _mm256_store_si256((__m256i *)(dqcoeff_ptr), zero);
#if CONFIG_VP9_HIGHBITDEPTH
      _mm256_store_si256((__m256i *)(qcoeff_ptr + 8), zero);
      _mm256_store_si256((__m256i *)(dqcoeff_ptr + 8), zero);
#endif  // CONFIG_VP9_HIGHBITDEPTH
      return;
    }
    calculate_qcoeff(&qcoeff0, round, quant, shift);

    // Reinsert signs
    qcoeff0 = _mm256_sign_epi16(qcoeff0, coeff0);

    // Mask out zbin threshold coeffs
    qcoeff0 = _mm256_and_si256(qcoeff0, cmp_mask0);

    store_tran_low(qcoeff0, qcoeff_ptr);

    coeff0 = calculate_dqcoeff(qcoeff0, dequant);

    store_tran_low(coeff0, dqcoeff_ptr);

    eob = scan_for_eob_single(&coeff0, cmp_mask0, iscan_ptr, 0, zero);
  }
  else {
    // Do DC and first 15 AC.
    coeff0 = load_tran_low(coeff_ptr);
    coeff1 = load_tran_low(coeff_ptr + 16);

    qcoeff0 = _mm256_abs_epi16(coeff0);
    qcoeff1 = _mm256_abs_epi16(coeff1);

    cmp_mask0 = _mm256_cmpgt_epi16(qcoeff0, zbin);
    zbin = _mm256_unpackhi_epi64(zbin, zbin);  // Switch DC to AC
    cmp_mask1 = _mm256_cmpgt_epi16(qcoeff1, zbin);

    all_zero = _mm256_or_si256(cmp_mask0, cmp_mask1);
    if (_mm256_testz_si256(all_zero, all_zero)) {
      _mm256_store_si256((__m256i *)(qcoeff_ptr), zero);
      _mm256_store_si256((__m256i *)(dqcoeff_ptr), zero);
      _mm256_store_si256((__m256i *)(qcoeff_ptr + 16), zero);
      _mm256_store_si256((__m256i *)(dqcoeff_ptr + 16), zero);
#if CONFIG_VP9_HIGHBITDEPTH
      _mm256_store_si256((__m256i *)(qcoeff_ptr + 8), zero);
      _mm256_store_si256((__m256i *)(dqcoeff_ptr + 8), zero);
      _mm256_store_si256((__m256i *)(qcoeff_ptr + 24), zero);
      _mm256_store_si256((__m256i *)(dqcoeff_ptr + 24), zero);
#endif  // CONFIG_VP9_HIGHBITDEPTH

      round = _mm256_unpackhi_epi64(round, round);
      quant = _mm256_unpackhi_epi64(quant, quant);
      shift = _mm256_unpackhi_epi64(shift, shift);
      dequant = _mm256_unpackhi_epi64(dequant, dequant);
    } else {
      calculate_qcoeff(&qcoeff0, round, quant, shift);
      round = _mm256_unpackhi_epi64(round, round);
      quant = _mm256_unpackhi_epi64(quant, quant);
      shift = _mm256_unpackhi_epi64(shift, shift);
      calculate_qcoeff(&qcoeff1, round, quant, shift);

      // Reinsert signs
      qcoeff0 = _mm256_sign_epi16(qcoeff0, coeff0);
      qcoeff1 = _mm256_sign_epi16(qcoeff1, coeff1);

      // Mask out zbin threshold coeffs
      qcoeff0 = _mm256_and_si256(qcoeff0, cmp_mask0);
      qcoeff1 = _mm256_and_si256(qcoeff1, cmp_mask1);

      store_tran_low(qcoeff0, qcoeff_ptr);
      store_tran_low(qcoeff1, qcoeff_ptr + 16);

      coeff0 = calculate_dqcoeff(qcoeff0, dequant);
      dequant = _mm256_unpackhi_epi64(dequant, dequant);
      coeff1 = calculate_dqcoeff(qcoeff1, dequant);

      store_tran_low(coeff0, dqcoeff_ptr);
      store_tran_low(coeff1, dqcoeff_ptr + 16);

      eob = scan_for_eob(&coeff0, &coeff1, cmp_mask0, cmp_mask1, iscan_ptr, 0,
                         zero);
    }
	}

  // AC only loop.
  for (index = 32; index < n_coeffs; index += 32) {
    coeff0 = load_tran_low(coeff_ptr + index);
    coeff1 = load_tran_low(coeff_ptr + index + 16);

    qcoeff0 = _mm256_abs_epi16(coeff0);
    qcoeff1 = _mm256_abs_epi16(coeff1);

    cmp_mask0 = _mm256_cmpgt_epi16(qcoeff0, zbin);
    cmp_mask1 = _mm256_cmpgt_epi16(qcoeff1, zbin);

    all_zero = _mm256_or_si256(cmp_mask0, cmp_mask1);
    if (_mm256_testz_si256(all_zero, all_zero)) {
      _mm256_store_si256((__m256i *)(qcoeff_ptr + index), zero);
      _mm256_store_si256((__m256i *)(dqcoeff_ptr + index), zero);
      _mm256_store_si256((__m256i *)(qcoeff_ptr + index + 16), zero);
      _mm256_store_si256((__m256i *)(dqcoeff_ptr + index + 16), zero);
#if CONFIG_VP9_HIGHBITDEPTH
      _mm256_store_si256((__m256i *)(qcoeff_ptr + index + 8), zero);
      _mm256_store_si256((__m256i *)(dqcoeff_ptr + index + 8), zero);
      _mm256_store_si256((__m256i *)(qcoeff_ptr + index + 24), zero);
      _mm256_store_si256((__m256i *)(dqcoeff_ptr + index + 24), zero);
#endif  // CONFIG_VP9_HIGHBITDEPTH
      continue;
    }

    calculate_qcoeff(&qcoeff0, round, quant, shift);
    calculate_qcoeff(&qcoeff1, round, quant, shift);

    qcoeff0 = _mm256_sign_epi16(qcoeff0, coeff0);
    qcoeff1 = _mm256_sign_epi16(qcoeff1, coeff1);

    qcoeff0 = _mm256_and_si256(qcoeff0, cmp_mask0);
    qcoeff1 = _mm256_and_si256(qcoeff1, cmp_mask1);

    store_tran_low(qcoeff0, qcoeff_ptr + index);
    store_tran_low(qcoeff1, qcoeff_ptr + index + 16);

    coeff0 = calculate_dqcoeff(qcoeff0, dequant);
    coeff1 = calculate_dqcoeff(qcoeff1, dequant);

    store_tran_low(coeff0, dqcoeff_ptr + index);
    store_tran_low(coeff1, dqcoeff_ptr + index + 16);

    eob0 = scan_for_eob(&coeff0, &coeff1, cmp_mask0, cmp_mask1, iscan_ptr,
                        index, zero);
    eob = _mm256_max_epi16(eob, eob0);
  }

  *eob_ptr = accumulate_eob(eob);
}
