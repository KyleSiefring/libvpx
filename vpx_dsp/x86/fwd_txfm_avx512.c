/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#define DCT_HIGH_BIT_DEPTH 0
#define FDCT8x8_2D vpx_fdct8x8_avx512
#include "vpx_dsp/x86/fwd_txfm_impl_avx512.h"
#undef FDCT4x4_2D
#undef FDCT8x8_2D
#undef FDCT16x16_2D

#if CONFIG_VP9_HIGHBITDEPTH
#define DCT_HIGH_BIT_DEPTH 1
#define FDCT8x8_2D vpx_highbd_fdct8x8_avx512
#include "vpx_dsp/x86/fwd_txfm_impl_avx512.h"  // NOLINT
#undef FDCT4x4_2D
#undef FDCT8x8_2D
#undef FDCT16x16_2D
#undef DCT_HIGH_BIT_DEPTH
#endif  // CONFIG_VP9_HIGHBITDEPTH
