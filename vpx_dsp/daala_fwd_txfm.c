/*
 * Copyright (c) 2017, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "vpx_dsp/daala_tx.h"
#include "vpx_dsp/daala_fwd_txfm.h"

typedef void (*daala_ftx)(od_coeff[], const od_coeff *, int);

static daala_ftx tx_map[TX_SIZES] = {od_bin_fdct4, od_bin_fdct8, od_bin_fdct16, od_bin_fdct32};

void daala_fwd_txfm(const int16_t *input_pixels,
                    tran_low_t *output_coeffs, int input_stride,
                    TX_SIZE tx_size) {
  //const int upshift = TX_COEFF_DEPTH - txfm_param->bd;
  const int upshift = 12 - 8;
  const int downshift = 1 + (tx_size == TX_32X32);
  const int cols = 4 << tx_size;
  const int rows = 4 << tx_size;
  daala_ftx tx = tx_map[tx_size];
  od_coeff tmp[MAX_TX_SIZE];
  int r;
  int c;
  // Transform columns
  for (c = 0; c < cols; ++c) {
    // Cast and shift
    for (r = 0; r < rows; ++r)
      tmp[r] =
          ((od_coeff)(input_pixels[r * input_stride + c])) * (1 << upshift);
    tx(tmp, tmp, 1);
    // No ystride in daala_tx lowlevel functions, store output vector
    // into column the long way
    
    for (r = 0; r < rows; ++r) output_coeffs[r * cols + c] = tmp[r] >> downshift;
  }
  // Transform rows
  for (r = 0; r < rows; ++r) {
    tx(output_coeffs + r * cols, output_coeffs + r * cols, 1);
  }
}
