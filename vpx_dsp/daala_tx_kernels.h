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

/* This header does not use an include guard.
   It is intentionally designed to be included multiple times.
   The file that includes it should define the following macros:

   OD_KERNEL  A label for the kernel. This should be unique for each inclusion
   OD_COEFF   The type of a coefficient or SIMD register, e.g., __m128i
   OD_ADD     The intrinsic function for addition
   OD_SUB     The intrinsic function for subtraction
   OD_RSHIFT1 The function that implements an unbiased right shift by 1
   OD_ADD_AVG The function that implements a signed PAVG[WD]
              I.e., (a + b + TX_AVG_BIAS) >> 1, without overflow
   OD_SUB_AVG The function that implements a VHRSUB.S<16|32>
              I.e., (a - b + TX_AVG_BIAS) >> 1, without overflow
   OD_MUL     The function that implements the multiplies
              I.e., (a * b + ((1 << r) >> 1)) >> r, without overflow
   OD_SWAP    The function that swaps two SIMD registers

   See daala_tx.c for a scalar example and x86/daala_inv_txfm_avx2.c for SIMD
   examples. */

#include "vpx_dsp/daala_tx.h"

#if !defined(OD_KERNEL_FUNC)
#define OD_KERNEL_FUNC_IMPL(name, kernel) name##_##kernel
#define OD_KERNEL_FUNC_WRAPPER(name, kernel) OD_KERNEL_FUNC_IMPL(name, kernel)
#define OD_KERNEL_FUNC(name) OD_KERNEL_FUNC_WRAPPER(name, OD_KERNEL)
#endif

/* clang-format off */

/* Two multiply rotation primative (used when rotating by Pi/4). */
static INLINE void OD_KERNEL_FUNC(od_rot2)(OD_COEFF *p0, OD_COEFF *p1,
 OD_COEFF t, int c0, int q0, int c1, int q1) {
  *p1 = OD_MUL(*p0, c0, q0);
  *p0 = OD_MUL(t, c1, q1);
}

/* Three multiply rotation primative. */
static INLINE void OD_KERNEL_FUNC(od_rot3)(OD_COEFF *p0, OD_COEFF *p1,
 OD_COEFF *t, OD_COEFF *u, int c0, int q0, int c1, int q1, int c2, int q2) {
  *u = OD_MUL(*p0, c0, q0);
  *p0 = OD_MUL(*p1, c1, q1);
  *t = OD_MUL(*t, c2, q2);
}

/* Rotate by Pi/4 and add. */
static INLINE void OD_KERNEL_FUNC(od_rotate_pi4_kernel)(OD_COEFF *p0,
 OD_COEFF *p1, int c0, int q0, int c1, int q1, int type, int avg) {
  OD_COEFF t;
  t = type == TX_ADD ?
   avg ? OD_ADD_AVG(*p1, *p0) : OD_ADD(*p1, *p0) :
   avg ? OD_SUB_AVG(*p1, *p0) : OD_SUB(*p1, *p0);
  OD_KERNEL_FUNC(od_rot2)(p0, p1, t, c0, q0, c1, q1);
  *p1 = type == TX_ADD ? OD_SUB(*p1, *p0) : OD_ADD(*p1, *p0);
}

#undef od_rotate_pi4_add
#define od_rotate_pi4_add(p0, p1, c0, q0, c1, q1) \
 OD_KERNEL_FUNC(od_rotate_pi4_kernel)(p0, p1, c0, q0, c1, q1, TX_ADD, TX_NONE)
#undef od_rotate_pi4_sub
#define od_rotate_pi4_sub(p0, p1, c0, q0, c1, q1) \
 OD_KERNEL_FUNC(od_rotate_pi4_kernel)(p0, p1, c0, q0, c1, q1, TX_SUB, TX_NONE)

#undef od_rotate_pi4_add_avg
#define od_rotate_pi4_add_avg(p0, p1, c0, q0, c1, q1) \
 OD_KERNEL_FUNC(od_rotate_pi4_kernel)(p0, p1, c0, q0, c1, q1, TX_ADD, TX_AVG)
#undef od_rotate_pi4_sub_avg
#define od_rotate_pi4_sub_avg(p0, p1, c0, q0, c1, q1) \
 OD_KERNEL_FUNC(od_rotate_pi4_kernel)(p0, p1, c0, q0, c1, q1, TX_SUB, TX_AVG)

/* Rotate and add. */
static INLINE void OD_KERNEL_FUNC(od_rotate_kernel)(OD_COEFF *p0, OD_COEFF *p1,
 OD_COEFF v, int c0, int q0, int c1, int q1, int c2, int q2,
 int type, int avg, int shift) {
  OD_COEFF u;
  OD_COEFF t;
  t = type == TX_ADD ?
   avg ? OD_ADD_AVG(*p1, v) : OD_ADD(*p1, v) :
   avg ? OD_SUB_AVG(*p1, v) : OD_SUB(*p1, v);
  OD_KERNEL_FUNC(od_rot3)(p0, p1, &t, &u, c0, q0, c1, q1, c2, q2);
  *p0 = OD_ADD(*p0, t);
  if (shift) t = OD_RSHIFT1(t);
  *p1 = type == TX_ADD ? OD_SUB(u, t) : OD_ADD(u, t);
}

#undef od_rotate_add
#define od_rotate_add(p0, p1, c0, q0, c1, q1, c2, q2, shift) \
 OD_KERNEL_FUNC(od_rotate_kernel)(p0, p1, *p0, c0, q0, c1, q1, c2, q2, \
                                  TX_ADD, TX_NONE, shift)
#undef od_rotate_sub
#define od_rotate_sub(p0, p1, c0, q0, c1, q1, c2, q2, shift) \
 OD_KERNEL_FUNC(od_rotate_kernel)(p0, p1, *p0, c0, q0, c1, q1, c2, q2, \
                                  TX_SUB, TX_NONE, shift)

#undef od_rotate_add_avg
#define od_rotate_add_avg(p0, p1, c0, q0, c1, q1, c2, q2, shift) \
 OD_KERNEL_FUNC(od_rotate_kernel)(p0, p1, *p0, c0, q0, c1, q1, c2, q2, \
                                  TX_ADD, TX_AVG, shift)
#undef od_rotate_sub_avg
#define od_rotate_sub_avg(p0, p1, c0, q0, c1, q1, c2, q2, shift) \
 OD_KERNEL_FUNC(od_rotate_kernel)(p0, p1, *p0, c0, q0, c1, q1, c2, q2, \
                                  TX_SUB, TX_AVG, shift)

#undef od_rotate_add_half
#define od_rotate_add_half(p0, p1, v, c0, q0, c1, q1, c2, q2, shift) \
 OD_KERNEL_FUNC(od_rotate_kernel)(p0, p1, v, c0, q0, c1, q1, c2, q2, \
                                  TX_ADD, TX_NONE, shift)
#undef od_rotate_sub_half
#define od_rotate_sub_half(p0, p1, v, c0, q0, c1, q1, c2, q2, shift) \
 OD_KERNEL_FUNC(od_rotate_kernel)(p0, p1, v, c0, q0, c1, q1, c2, q2, \
                                  TX_SUB, TX_NONE, shift)

/* Rotate and subtract with negation. */
static INLINE void OD_KERNEL_FUNC(od_rotate_neg_kernel)(OD_COEFF *p0,
 OD_COEFF *p1, int c0, int q0, int c1, int q1, int c2, int q2, int avg) {
  OD_COEFF u;
  OD_COEFF t;
  t = avg ? OD_SUB_AVG(*p0, *p1) : OD_SUB(*p0, *p1);
  OD_KERNEL_FUNC(od_rot3)(p0, p1, &t, &u, c0, q0, c1, q1, c2, q2);
  *p0 = OD_SUB(*p0, t);
  *p1 = OD_SUB(t, u);
}

#undef od_rotate_neg
#define od_rotate_neg(p0, p1, c0, q0, c1, q1, c2, q2) \
 OD_KERNEL_FUNC(od_rotate_neg_kernel)(p0, p1, c0, q0, c1, q1, c2, q2, TX_NONE)
#undef od_rotate_neg_avg
#define od_rotate_neg_avg(p0, p1, c0, q0, c1, q1, c2, q2) \
 OD_KERNEL_FUNC(od_rotate_neg_kernel)(p0, p1, c0, q0, c1, q1, c2, q2, TX_AVG)

/* Computes the +/- addition butterfly (asymmetric output).
   The inverse to this function is od_butterfly_add_asym().

    p0 = p0 + p1;
    p1 = p1 - p0/2; */
static INLINE void OD_KERNEL_FUNC(od_butterfly_add)(OD_COEFF *p0,
 OD_COEFF *p0h, OD_COEFF *p1) {
  OD_COEFF p0h_;
  *p0 = OD_ADD(*p0, *p1);
  p0h_ = OD_RSHIFT1(*p0);
  *p1 = OD_SUB(*p1, p0h_);
  if (p0h != NULL) *p0h = p0h_;
}

/* Computes the asymmetric +/- addition butterfly (unscaled output).
   The inverse to this function is od_butterfly_add().

    p1 = p1 + p0/2;
    p0 = p0 - p1; */
static INLINE void OD_KERNEL_FUNC(od_butterfly_add_asym)(OD_COEFF *p0,
 OD_COEFF p0h, OD_COEFF *p1) {
  *p1 = OD_ADD(*p1, p0h);
  *p0 = OD_SUB(*p0, *p1);
}

/* Computes the +/- subtraction butterfly (asymmetric output).
   The inverse to this function is od_butterfly_sub_asym().

    p0 = p0 - p1;
    p1 = p1 + p0/2; */
static INLINE void OD_KERNEL_FUNC(od_butterfly_sub)(OD_COEFF *p0,
 OD_COEFF *p0h, OD_COEFF *p1) {
  OD_COEFF p0h_;
  *p0 = OD_SUB(*p0, *p1);
  p0h_ = OD_RSHIFT1(*p0);
  *p1 = OD_ADD(*p1, p0h_);
  if (p0h != NULL) *p0h = p0h_;
}

/* Computes the asymmetric +/- subtraction butterfly (unscaled output).
   The inverse to this function is od_butterfly_sub().

    p1 = p1 - p0/2;
    p0 = p0 + p1; */
static INLINE void OD_KERNEL_FUNC(od_butterfly_sub_asym)(OD_COEFF *p0,
 OD_COEFF p0h, OD_COEFF *p1) {
  *p1 = OD_SUB(*p1, p0h);
  *p0 = OD_ADD(*p0, *p1);
}

/* Computes the +/- subtract and negate butterfly (asymmetric output).
   The inverse to this function is od_butterfly_neg_asym().

    p1 = p1 - p0;
    p0 = p0 + p1/2;
    p1 = -p1; */
static INLINE void OD_KERNEL_FUNC(od_butterfly_neg)(OD_COEFF *p0, OD_COEFF *p1, OD_COEFF *p1h) {
  *p1 = OD_SUB(*p0, *p1);
  *p1h = OD_RSHIFT1(*p1);
  *p0 = OD_SUB(*p0, *p1h);
}

/*  Computes the asymmetric +/- negate and subtract butterfly (unscaled output).
    The inverse to this function is od_butterfly_neg().

    p1 = -p1;
    p0 = p0 - p1/2;
    p1 = p1 + p0; */
static INLINE void OD_KERNEL_FUNC(od_butterfly_neg_asym)(OD_COEFF *p0,
 OD_COEFF *p1, OD_COEFF p1h) {
  *p0 = OD_ADD(*p0, p1h);
  *p1 = OD_SUB(*p0, *p1);
}

/* --- 2-point Transforms --- */

/**
 * 2-point orthonormal Type-II fDCT
 */
static INLINE void OD_KERNEL_FUNC(od_fdct_2)(OD_COEFF *p0, OD_COEFF *p1) {
  /* 11585/8192 = Sin[Pi/4] + Cos[Pi/4]  = 1.4142135623730951 */
  /* 11585/8192 = 2*Cos[Pi/4]            = 1.4142135623730951 */
  od_rotate_pi4_sub_avg(p1, p0, 11585, 13, 11585, 13);
}

/**
 * 2-point orthonormal Type-II iDCT
 */
static INLINE void OD_KERNEL_FUNC(od_idct_2)(OD_COEFF *p0, OD_COEFF *p1) {
  /*  11585/8192 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* 11585/16384 = Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(p0, p1, 11585, 13, 11585, 14);
}

/**
 * 2-point asymmetric Type-II fDCT
 */
static INLINE void OD_KERNEL_FUNC(od_fdct_2_asym)(OD_COEFF *p0, OD_COEFF *p1,
 OD_COEFF p1h) {
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(p0, p1, p1h);
}

/**
 * 2-point asymmetric Type-II iDCT
 */
static INLINE void OD_KERNEL_FUNC(od_idct_2_asym)(OD_COEFF *p0, OD_COEFF *p1,
 OD_COEFF *p1h) {
  OD_KERNEL_FUNC(od_butterfly_neg)(p0, p1, p1h);
}

/**
 * 2-point orthonormal Type-IV fDCT
 */
static INLINE void OD_KERNEL_FUNC(od_fdst_2)(OD_COEFF *p0, OD_COEFF *p1) {

  /* Stage 0 */

  /* 10703/8192 = Sin[3*Pi/8] + Cos[3*Pi/8]  = 1.3065629648763766 */
  /* 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8]  = 0.5411961001461971 */
  /*  3135/4096 = 2*Cos[3*Pi/8]              = 0.7653668647301796 */
  od_rotate_add_avg(p0, p1, 10703, 13, 8867, 14, 3135, 12, TX_NONE);
}

/**
 * 2-point orthonormal Type-IV iDCT
 */
static INLINE void OD_KERNEL_FUNC(od_idst_2)(OD_COEFF *p0, OD_COEFF *p1) {
  OD_KERNEL_FUNC(od_fdst_2)(p0, p1);
}

/**
 * 2-point asymmetric Type-IV fDCT
 */
static INLINE void OD_KERNEL_FUNC(od_fdst_2_asym)(OD_COEFF *p0, OD_COEFF p0h,
 OD_COEFF *p1) {

  /* Stage 0 */

  /*   473/512 = (Sin[3*Pi/8] + Cos[3*Pi/8])/Sqrt[2] = 0.9238795325112867 */
  /* 3135/4096 = (Sin[3*Pi/8] - Cos[3*Pi/8])*Sqrt[2] = 0.7653668647301795 */
  /* 4433/8192 = Cos[3*Pi/8]*Sqrt[2]                 = 0.5411961001461971 */
  od_rotate_add_half(p0, p1, p0h, 473, 9, 3135, 12, 4433, 13, TX_NONE);
}

/**
 * 2-point asymmetric Type-IV iDCT
 */
static INLINE void OD_KERNEL_FUNC(od_idst_2_asym)(OD_COEFF *p0, OD_COEFF *p1) {

  /* Stage 0 */

  /*   473/512 = (Sin[3*Pi/8] + Cos[3*Pi/8])/Sqrt[2] = 0.9238795325112867 */
  /* 3135/4096 = (Sin[3*Pi/8] - Cos[3*Pi/8])*Sqrt[2] = 0.7653668647301795 */
  /* 8867/8192 = 2*Cos[3*Pi/8]*Sqrt[2]               = 1.0823922002923940 */
  od_rotate_add_avg(p0, p1, 473, 9, 3135, 12, 8867, 13, TX_SHIFT);
}

/* --- 4-point Transforms --- */

/**
 * 4-point orthonormal Type-II fDCT
 */
static INLINE void OD_KERNEL_FUNC(od_fdct_4)(OD_COEFF *q0, OD_COEFF *q1,
 OD_COEFF *q2, OD_COEFF *q3) {
  OD_COEFF q1h;
  OD_COEFF q3h;

  /* +/- Butterflies with asymmetric output. */
  OD_KERNEL_FUNC(od_butterfly_neg)(q0, q3, &q3h);
  OD_KERNEL_FUNC(od_butterfly_add)(q1, &q1h, q2);

  /* Embedded 2-point transforms with asymmetric input. */
  OD_KERNEL_FUNC(od_fdct_2_asym)(q0, q1, q1h);
  OD_KERNEL_FUNC(od_fdst_2_asym)(q3, q3h, q2);
}

/**
 * 4-point orthonormal Type-II iDCT
 */
static INLINE void OD_KERNEL_FUNC(od_idct_4)(OD_COEFF *q0, OD_COEFF *q2,
                                             OD_COEFF *q1, OD_COEFF *q3)  {
  OD_COEFF q1h;

  /* Embedded 2-point transforms with asymmetric output. */
  OD_KERNEL_FUNC(od_idst_2_asym)(q3, q2);
  OD_KERNEL_FUNC(od_idct_2_asym)(q0, q1, &q1h);

  /* +/- Butterflies with asymmetric input. */
  OD_KERNEL_FUNC(od_butterfly_add_asym)(q1, q1h, q2);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(q0, q3, OD_RSHIFT1(*q3));
}

/**
 * 4-point asymmetric Type-II fDCT
 */
static INLINE void OD_KERNEL_FUNC(od_fdct_4_asym)(OD_COEFF *q0, OD_COEFF *q1,
 OD_COEFF q1h, OD_COEFF *q2, OD_COEFF *q3, OD_COEFF q3h) {

  /* +/- Butterflies with asymmetric input. */
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(q0, q3, q3h);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(q1, q1h, q2);

  /* Embedded 2-point orthonormal transforms. */
  OD_KERNEL_FUNC(od_fdct_2)(q0, q1);
  OD_KERNEL_FUNC(od_fdst_2)(q3, q2);
}

/**
 * 4-point asymmetric Type-II iDCT
 */
static INLINE void OD_KERNEL_FUNC(od_idct_4_asym)(OD_COEFF *q0, OD_COEFF *q2,
 OD_COEFF *q1, OD_COEFF *q1h, OD_COEFF *q3, OD_COEFF *q3h)  {

  /* Embedded 2-point orthonormal transforms. */
  OD_KERNEL_FUNC(od_idst_2)(q3, q2);
  OD_KERNEL_FUNC(od_idct_2)(q0, q1);

  /* +/- Butterflies with asymmetric output. */
  OD_KERNEL_FUNC(od_butterfly_sub)(q1, q1h, q2);
  OD_KERNEL_FUNC(od_butterfly_neg)(q0, q3, q3h);
}

/**
 * 4-point orthonormal Type-IV fDST
 */
static INLINE void OD_KERNEL_FUNC(od_fdst_4)(OD_COEFF *q0, OD_COEFF *q1,
                                             OD_COEFF *q2, OD_COEFF *q3) {

  /* Stage 0 */

  /* 13623/16384 = (Sin[7*Pi/16] + Cos[7*Pi/16])/Sqrt[2] = 0.831469612302545 */
  /*   4551/4096 = (Sin[7*Pi/16] - Cos[7*Pi/16])*Sqrt[2] = 1.111140466039204 */
  /*  9041/32768 = Cos[7*Pi/16]*Sqrt[2]                  = 0.275899379282943 */
  od_rotate_add(q0, q3, 13623, 14, 4551, 12, 565, 11, TX_SHIFT);

  /* 16069/16384 = (Sin[5*Pi/16] + Cos[5*Pi/16])/Sqrt[2] = 0.9807852804032304 */
  /* 12785/32768 = (Sin[5*Pi/16] - Cos[5*Pi/16])*Sqrt[2] = 0.3901806440322566 */
  /*   1609/2048 = Cos[5*Pi/16]*Sqrt[2]                  = 0.7856949583871021 */
  od_rotate_sub(q2, q1, 16069, 14, 12785, 15, 1609, 11, TX_SHIFT);

  /* Stage 1 */

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(q0, OD_RSHIFT1(*q0), q1);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(q2, OD_RSHIFT1(*q2), q3);

  /* Stage 2 */

  /*  5793/4096 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951 */
  od_rotate_pi4_add_avg(q2, q1, 5793, 12, 11585, 13);
}

/**
 * 4-point orthonormal Type-IV iDST
 */
static INLINE void OD_KERNEL_FUNC(od_idst_4)(OD_COEFF *q0, OD_COEFF *q2,
                                             OD_COEFF *q1, OD_COEFF *q3) {
  OD_COEFF q0h;
  OD_COEFF q2h;

  /* Stage 0 */

  /*  5793/4096 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951 */
  od_rotate_pi4_add_avg(q2, q1, 5793, 12, 11585, 13);

  /* Stage 1 */

  OD_KERNEL_FUNC(od_butterfly_sub)(q2, &q2h, q3);
  OD_KERNEL_FUNC(od_butterfly_sub)(q0, &q0h, q1);

  /* Stage 2 */

  /*   8035/8192 = (Sin[5*Pi/16] + Cos[5*Pi/16])/Sqrt[2] = 0.9807852804032304 */
  /* 12785/32768 = (Sin[5*Pi/16] - Cos[5*Pi/16])*Sqrt[2] = 0.3901806440322566 */
  /* 12873/16384 = Cos[5*Pi/16]*Sqrt[2]                  = 0.7856949583871021 */
  od_rotate_sub_half(q2, q1, q2h, 8035, 13, 12785, 15, 12873, 14, TX_NONE);

  /*   6811/8192 = (Sin[7*Pi/16] + Cos[7*Pi/16])/Sqrt[2] = 0.831469612302545 */
  /* 18205/16384 = (Sin[7*Pi/16] - Cos[7*Pi/16])*Sqrt[2] = 1.111140466039204 */
  /*  9041/32768 = Cos[7*Pi/16]*Sqrt[2]                  = 0.275899379282943 */
  od_rotate_add_half(q0, q3, q0h, 6811, 13, 18205, 14, 9041, 15, TX_NONE);
}

/**
 * 4-point asymmetric Type-IV fDST
 */
static INLINE void OD_KERNEL_FUNC(od_fdst_4_asym)(OD_COEFF *q0, OD_COEFF q0h,
 OD_COEFF *q1, OD_COEFF *q2, OD_COEFF q2h, OD_COEFF *q3) {

  /* Stage 0 */

  /*  9633/16384 = (Sin[7*Pi/16] + Cos[7*Pi/16])/2 = 0.5879378012096793 */
  /*  12873/8192 = (Sin[7*Pi/16] - Cos[7*Pi/16])*2 = 1.5713899167742045 */
  /* 12785/32768 = Cos[7*Pi/16]*2                  = 0.3901806440322565 */
  od_rotate_add_half(q0, q3, q0h, 9633, 14, 12873, 13, 12785, 15, TX_SHIFT);

  /* 11363/16384 = (Sin[5*Pi/16] + Cos[5*Pi/16])/2 = 0.6935199226610738 */
  /* 18081/32768 = (Sin[5*Pi/16] - Cos[5*Pi/16])*2 = 0.5517987585658861 */
  /*  4551/4096 = Cos[5*Pi/16]*2                  = 1.1111404660392044 */
  od_rotate_sub_half(q2, q1, q2h, 11363, 14, 18081, 15, 4551, 12, TX_SHIFT);

  /* Stage 1 */

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(q0, OD_RSHIFT1(*q0), q1);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(q2, OD_RSHIFT1(*q2), q3);

  /* Stage 2 */

  /* 11585/8192 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951 */
  od_rotate_pi4_add_avg(q2, q1, 11585, 13, 11585, 13);
}

/**
 * 4-point asymmetric Type-IV iDST
 */
static INLINE void OD_KERNEL_FUNC(od_idst_4_asym)(OD_COEFF *q0, OD_COEFF *q2,
                                                  OD_COEFF *q1, OD_COEFF *q3) {
  OD_COEFF q0h;
  OD_COEFF q2h;

  /* Stage 0 */

  /* 11585/8192 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951 */
  od_rotate_pi4_add_avg(q2, q1, 11585, 13, 11585, 13);

  /* Stage 1 */

  OD_KERNEL_FUNC(od_butterfly_sub)(q2, &q2h, q3);
  OD_KERNEL_FUNC(od_butterfly_sub)(q0, &q0h, q1);

  /* Stage 2 */

  /* 11363/16384 = (Sin[5*Pi/16] + Cos[5*Pi/16])/2 = 0.6935199226610738 */
  /* 18081/32768 = (Sin[5*Pi/16] - Cos[5*Pi/16])*2 = 0.5517987585658861 */
  /* 18205/16384 = Cos[5*Pi/16]*2                  = 1.1111404660392044 */
  od_rotate_sub_half(q2, q1, q2h, 11363, 14, 18081, 15, 18205, 14, TX_SHIFT);

  /*  9633/16384 = (Sin[7*Pi/16] + Cos[7*Pi/16])/2 = 0.5879378012096793 */
  /*  12873/8192 = (Sin[7*Pi/16] - Cos[7*Pi/16])*2 = 1.5713899167742045 */
  /* 12785/32768 = Cos[7*Pi/16]*2                  = 0.3901806440322565 */
  od_rotate_add_half(q0, q3, q0h, 9633, 14, 12873, 13, 12785, 15, TX_SHIFT);
}

/**
 * 4-point asymmetric Type-VII fDST
 */
static INLINE void OD_KERNEL_FUNC(od_fdst_vii_4)(OD_COEFF *q0, OD_COEFF *q1,
                                                 OD_COEFF *q2, OD_COEFF *q3) {
  /* 11 adds, 5 "muls", 2 shifts.*/
  OD_COEFF t0;
  OD_COEFF t1;
  OD_COEFF t2;
  OD_COEFF t3;
  OD_COEFF t3h;
  OD_COEFF t4;
  OD_COEFF u4;
  t0 = OD_ADD(*q1, *q3);
  t1 = OD_ADD(*q1, OD_SUB_AVG(*q0, t0));
  t2 = OD_SUB(*q0, *q1);
  t3 = *q2;
  t4 = OD_ADD(*q0, *q3);
  /* 7021/16384 ~= 2*Sin[2*Pi/9]/3 ~= 0.428525073124360 */
  t0 = OD_MUL(t0, 7021, 14);
  /* 37837/32768 ~= 4*Sin[3*Pi/9]/3 ~= 1.154700538379252 */
  t1 = OD_MUL(t1, 37837, 15);
  /* 21513/32768 ~= 2*Sin[4*Pi/9]/3 ~= 0.656538502008139 */
  t2 = OD_MUL(t2, 21513, 15);
  /* 37837/32768 ~= 4*Sin[3*Pi/9]/3 ~= 1.154700538379252 */
  t3 = OD_MUL(t3, 37837, 15);
  /* 467/2048 ~= 2*Sin[1*Pi/9]/3 ~= 0.228013428883779 */
  t4 = OD_MUL(t4, 467, 11);
  t3h = OD_RSHIFT1(t3);
  u4 = OD_ADD(t4, t3h);
  *q0 = OD_ADD(t0, u4);
  *q2 = t1;
  *q1 = OD_ADD(t0, OD_SUB(t2, t3h));
  *q3 = OD_ADD(t2, OD_SUB(t3, u4));
}

/**
 * 4-point asymmetric Type-VII iDST
 */
static INLINE void OD_KERNEL_FUNC(od_idst_vii_4)(OD_COEFF *q0, OD_COEFF *q2,
                                                 OD_COEFF *q1, OD_COEFF *q3) {
  /* 11 adds, 5 "muls", 2 shifts.*/
  OD_COEFF t0;
  OD_COEFF t1;
  OD_COEFF t2;
  OD_COEFF t3;
  OD_COEFF t3h;
  OD_COEFF t4;
  OD_COEFF u4;
  t0 = OD_SUB(*q0, *q3);
  t1 = OD_ADD(*q0, *q1);
  t2 = OD_ADD(*q3, OD_SUB_AVG(t0, *q1));
  t3 = *q2;
  t4 = OD_ADD(*q1, *q3);
  /* 467/2048 ~= 2*Sin[1*Pi/9]/3 ~= 0.228013428883779 */
  t0 = OD_MUL(t0, 467, 11);
  /* 7021/16384 ~= 2*Sin[2*Pi/9]/3 ~= 0.428525073124360 */
  t1 = OD_MUL(t1, 7021, 14);
  /* 37837/32768 ~= 4*Sin[3*Pi/9]/3 ~= 1.154700538379252 */
  t2 = OD_MUL(t2, 37837, 15);
  /* 37837/32768 ~= 4*Sin[3*Pi/9]/3 ~= 1.154700538379252 */
  t3 = OD_MUL(t3, 37837, 15);
  /* 21513/32768 ~= 2*Sin[4*Pi/9]/3 ~= 0.656538502008139 */
  t4 = OD_MUL(t4, 21513, 15);
  t3h = OD_RSHIFT1(t3);
  u4 = OD_ADD(t4, t3h);
  *q0 = OD_ADD(t0, u4);
  *q2 = t2;
  *q1 = OD_ADD(t1, OD_SUB(t3, u4));
  *q3 = OD_ADD(t0, OD_SUB(t1, t3h));
}

static INLINE void OD_KERNEL_FUNC(od_flip_idst_vii_4)(OD_COEFF *q0,
 OD_COEFF *q2, OD_COEFF *q1, OD_COEFF *q3) {
  OD_KERNEL_FUNC(od_idst_vii_4)(q0, q2, q1, q3);
  OD_SWAP(q0, q3);
  OD_SWAP(q2, q1);
}

/* --- 8-point Transforms --- */

/**
 * 8-point orthonormal Type-II fDCT
 */
static INLINE void OD_KERNEL_FUNC(od_fdct_8)(OD_COEFF *r0, OD_COEFF *r1,
                                             OD_COEFF *r2, OD_COEFF *r3,
                                             OD_COEFF *r4, OD_COEFF *r5,
                                             OD_COEFF *r6, OD_COEFF *r7) {
  OD_COEFF r1h;
  OD_COEFF r3h;
  OD_COEFF r5h;
  OD_COEFF r7h;

  /* +/- Butterflies with asymmetric output. */
  OD_KERNEL_FUNC(od_butterfly_neg)(r0, r7, &r7h);
  OD_KERNEL_FUNC(od_butterfly_add)(r1, &r1h, r6);
  OD_KERNEL_FUNC(od_butterfly_neg)(r2, r5, &r5h);
  OD_KERNEL_FUNC(od_butterfly_add)(r3, &r3h, r4);

  /* Embedded 4-point forward transforms with asymmetric input. */
  OD_KERNEL_FUNC(od_fdct_4_asym)(r0, r1, r1h, r2, r3, r3h);
  OD_KERNEL_FUNC(od_fdst_4_asym)(r7, r7h, r6, r5, r5h, r4);
}

/**
 * 8-point orthonormal Type-II iDCT
 */
static INLINE void OD_KERNEL_FUNC(od_idct_8)(OD_COEFF *r0, OD_COEFF *r4,
                                             OD_COEFF *r2, OD_COEFF *r6,
                                             OD_COEFF *r1, OD_COEFF *r5,
                                             OD_COEFF *r3, OD_COEFF *r7) {
  OD_COEFF r1h;
  OD_COEFF r3h;

  /* Embedded 4-point inverse transforms with asymmetric output. */
  OD_KERNEL_FUNC(od_idst_4_asym)(r7, r5, r6, r4);
  OD_KERNEL_FUNC(od_idct_4_asym)(r0, r2, r1, &r1h, r3, &r3h);

  /* +/- Butterflies with asymmetric input. */
  OD_KERNEL_FUNC(od_butterfly_add_asym)(r3, r3h, r4);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(r2, r5, OD_RSHIFT1(*r5));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(r1, r1h, r6);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(r0, r7, OD_RSHIFT1(*r7));
}

/**
 * 8-point asymmetric Type-II fDCT
 */
static INLINE void OD_KERNEL_FUNC(od_fdct_8_asym)(OD_COEFF *r0, OD_COEFF *r1,
 OD_COEFF r1h, OD_COEFF *r2, OD_COEFF *r3, OD_COEFF r3h, OD_COEFF *r4,
 OD_COEFF *r5, OD_COEFF r5h, OD_COEFF *r6, OD_COEFF *r7, OD_COEFF r7h) {

  /* +/- Butterflies with asymmetric input. */
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(r0, r7, r7h);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(r1, r1h, r6);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(r2, r5, r5h);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(r3, r3h, r4);

  /* Embedded 4-point orthonormal transforms. */
  OD_KERNEL_FUNC(od_fdct_4)(r0, r1, r2, r3);
  OD_KERNEL_FUNC(od_fdst_4)(r7, r6, r5, r4);
}

/**
 * 8-point asymmetric Type-II iDCT
 */
static INLINE void OD_KERNEL_FUNC(od_idct_8_asym)(OD_COEFF *r0, OD_COEFF *r4,
 OD_COEFF *r2, OD_COEFF *r6, OD_COEFF *r1, OD_COEFF *r1h, OD_COEFF *r5,
 OD_COEFF *r5h, OD_COEFF *r3, OD_COEFF *r3h, OD_COEFF *r7, OD_COEFF *r7h)  {

  /* Embedded 4-point inverse orthonormal transforms. */
  OD_KERNEL_FUNC(od_idst_4)(r7, r5, r6, r4);
  OD_KERNEL_FUNC(od_idct_4)(r0, r2, r1, r3);

  /* +/- Butterflies with asymmetric output. */
  OD_KERNEL_FUNC(od_butterfly_sub)(r3, r3h, r4);
  OD_KERNEL_FUNC(od_butterfly_neg)(r2, r5, r5h);
  OD_KERNEL_FUNC(od_butterfly_sub)(r1, r1h, r6);
  OD_KERNEL_FUNC(od_butterfly_neg)(r0, r7, r7h);
}

/**
 * 8-point orthonormal Type-IV fDST
 */
static INLINE void OD_KERNEL_FUNC(od_fdst_8)(OD_COEFF *r0, OD_COEFF *r1,
                                             OD_COEFF *r2, OD_COEFF *r3,
                                             OD_COEFF *r4, OD_COEFF *r5,
                                             OD_COEFF *r6, OD_COEFF *r7) {
  OD_COEFF r0h;
  OD_COEFF r2h;
  OD_COEFF r5h;
  OD_COEFF r7h;

  /* Stage 0 */

  /* 17911/16384 = Sin[15*Pi/32] + Cos[15*Pi/32] = 1.0932018670017576 */
  /* 14699/16384 = Sin[15*Pi/32] - Cos[15*Pi/32] = 0.8971675863426363 */
  /*    803/8192 = Cos[15*Pi/32]                 = 0.0980171403295606 */
  od_rotate_add(r0, r7, 17911, 14, 14699, 14, 803, 13, TX_NONE);

  /* 20435/16384 = Sin[13*Pi/32] + Cos[13*Pi/32] = 1.24722501298667123 */
  /* 21845/32768 = Sin[13*Pi/32] - Cos[13*Pi/32] = 0.66665565847774650 */
  /*   1189/4096 = Cos[13*Pi/32]                 = 0.29028467725446233 */
  od_rotate_sub(r6, r1, 20435, 14, 21845, 15, 1189, 12, TX_NONE);

  /* 22173/16384 = Sin[11*Pi/32] + Cos[11*Pi/32] = 1.3533180011743526 */
  /*   3363/8192 = Sin[11*Pi/32] - Cos[11*Pi/32] = 0.4105245275223574 */
  /* 15447/32768 = Cos[11*Pi/32]                 = 0.47139673682599764 */
  od_rotate_add(r2, r5, 22173, 14, 3363, 13, 15447, 15, TX_NONE);

  /* 23059/16384 = Sin[9*Pi/32] + Cos[9*Pi/32] = 1.4074037375263826 */
  /*  2271/16384 = Sin[9*Pi/32] - Cos[9*Pi/32] = 0.1386171691990915 */
  /*   5197/8192 = Cos[9*Pi/32]                = 0.6343932841636455 */
  od_rotate_sub(r4, r3, 23059, 14, 2271, 14, 5197, 13, TX_NONE);

  /* Stage 1 */

  OD_KERNEL_FUNC(od_butterfly_add)(r0, &r0h, r3);
  OD_KERNEL_FUNC(od_butterfly_sub)(r2, &r2h, r1);
  OD_KERNEL_FUNC(od_butterfly_add)(r5, &r5h, r6);
  OD_KERNEL_FUNC(od_butterfly_sub)(r7, &r7h, r4);

  /* Stage 2 */

  OD_KERNEL_FUNC(od_butterfly_add_asym)(r7, r7h, r6);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(r5, r5h, r3);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(r2, r2h, r4);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(r0, r0h, r1);

  /* Stage 3 */

  /* 10703/8192 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /*  3135/4096 = 2*Cos[3*Pi/8]             = 0.7653668647301796 */
  od_rotate_sub_avg(r3, r4, 10703, 13, 8867, 14, 3135, 12, TX_NONE);

  /* 10703/8192 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /*  3135/4096 = 2*Cos[3*Pi/8]             = 0.7653668647301796 */
  od_rotate_neg_avg(r2, r5, 10703, 13, 8867, 14, 3135, 12);

  /* 11585/8192 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951 */
  od_rotate_pi4_sub_avg(r1, r6, 11585, 13, 11585, 13);
}

/**
 * 8-point orthonormal Type-IV iDST
 */
static INLINE void OD_KERNEL_FUNC(od_idst_8)(OD_COEFF *r0, OD_COEFF *r4,
                                             OD_COEFF *r2, OD_COEFF *r6,
                                             OD_COEFF *r1, OD_COEFF *r5,
                                             OD_COEFF *r3, OD_COEFF *r7) {
  OD_COEFF r0h;
  OD_COEFF r2h;
  OD_COEFF r5h;
  OD_COEFF r7h;

  /* Stage 3 */

  /* 11585/8192 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951 */
  od_rotate_pi4_add_avg(r6, r1, 11585, 13, 11585, 13);

  /* 10703/8192 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /*  3135/4096 = 2*Cos[3*Pi/8]             = 0.7653668647301796 */
  od_rotate_neg_avg(r5, r2, 10703, 13, 8867, 14, 3135, 12);

  /* 10703/8192 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /*  3135/4096 = 2*Cos[3*Pi/8]             = 0.7653668647301796 */
  od_rotate_add_avg(r4, r3, 10703, 13, 8867, 14, 3135, 12, TX_NONE);

  /* Stage 2 */

  OD_KERNEL_FUNC(od_butterfly_sub)(r0, &r0h, r1);
  OD_KERNEL_FUNC(od_butterfly_add)(r2, &r2h, r4);
  OD_KERNEL_FUNC(od_butterfly_add)(r5, &r5h, r3);
  OD_KERNEL_FUNC(od_butterfly_add)(r7, &r7h, r6);

  /* Stage 1 */

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(r7, r7h, r4);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(r5, r5h, r6);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(r2, r2h, r1);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(r0, r0h, r3);

  /* Stage 0 */

  /* 23059/16384 = Sin[9*Pi/32] + Cos[9*Pi/32] = 1.4074037375263826 */
  /*  2271/16384 = Sin[9*Pi/32] - Cos[9*Pi/32] = 0.1386171691990915 */
  /*   5197/8192 = Cos[9*Pi/32]                = 0.6343932841636455 */
  od_rotate_sub(r4, r3, 23059, 14, 2271, 14, 5197, 13, TX_NONE);

  /* 22173/16384 = Sin[11*Pi/32] + Cos[11*Pi/32] = 1.3533180011743526 */
  /*   3363/8192 = Sin[11*Pi/32] - Cos[11*Pi/32] = 0.4105245275223574 */
  /* 15447/32768 = Cos[11*Pi/32]                 = 0.47139673682599764 */
  od_rotate_add(r2, r5, 22173, 14, 3363, 13, 15447, 15, TX_NONE);

  /* 20435/16384 = Sin[13*Pi/32] + Cos[13*Pi/32] = 1.24722501298667123 */
  /* 21845/32768 = Sin[13*Pi/32] - Cos[13*Pi/32] = 0.66665565847774650 */
  /*   1189/4096 = Cos[13*Pi/32]                 = 0.29028467725446233 */
  od_rotate_sub(r6, r1, 20435, 14, 21845, 15, 1189, 12, TX_NONE);

  /* 17911/16384 = Sin[15*Pi/32] + Cos[15*Pi/32] = 1.0932018670017576 */
  /* 14699/16384 = Sin[15*Pi/32] - Cos[15*Pi/32] = 0.8971675863426363 */
  /*    803/8192 = Cos[15*Pi/32]                 = 0.0980171403295606 */
  od_rotate_add(r0, r7, 17911, 14, 14699, 14, 803, 13, TX_NONE);
}

static INLINE void OD_KERNEL_FUNC(od_flip_idst_8)(OD_COEFF *r0, OD_COEFF *r4,
                                                  OD_COEFF *r2, OD_COEFF *r6,
                                                  OD_COEFF *r1, OD_COEFF *r5,
                                                  OD_COEFF *r3, OD_COEFF *r7) {
  OD_KERNEL_FUNC(od_idst_8)(r0, r4, r2, r6, r1, r5, r3, r7);
  OD_SWAP(r0, r7);
  OD_SWAP(r4, r3);
  OD_SWAP(r2, r5);
  OD_SWAP(r6, r1);
}

/**
 * 8-point asymmetric Type-IV fDST
 */
static INLINE void OD_KERNEL_FUNC(od_fdst_8_asym)(OD_COEFF *r0, OD_COEFF r0h,
 OD_COEFF *r1, OD_COEFF *r2, OD_COEFF r2h, OD_COEFF *r3, OD_COEFF *r4,
 OD_COEFF r4h, OD_COEFF *r5, OD_COEFF *r6, OD_COEFF r6h, OD_COEFF *r7) {
  OD_COEFF r5h;
  OD_COEFF r7h;

  /* Stage 0 */

  /* 12665/16384 = (Sin[15*Pi/32] + Cos[15*Pi/32])/Sqrt[2] = 0.77301045336274 */
  /*   5197/4096 = (Sin[15*Pi/32] - Cos[15*Pi/32])*Sqrt[2] = 1.26878656832729 */
  /*  2271/16384 = Cos[15*Pi/32]*Sqrt[2]                   = 0.13861716919909 */
  od_rotate_add_half(r0, r7, r0h, 12665, 14, 5197, 12, 2271, 14, TX_NONE);

  /* 14449/16384 = Sin[13*Pi/32] + Cos[13*Pi/32])/Sqrt[2] = 0.881921264348355 */
  /* 30893/32768 = Sin[13*Pi/32] - Cos[13*Pi/32])*Sqrt[2] = 0.942793473651995 */
  /*   3363/8192 = Cos[13*Pi/32]*Sqrt[2]                  = 0.410524527522357 */
  od_rotate_sub_half(r6, r1, r6h, 14449, 14, 30893, 15, 3363, 13, TX_NONE);

  /* 15679/16384 = Sin[11*Pi/32] + Cos[11*Pi/32])/Sqrt[2] = 0.956940335732209 */
  /*   1189/2048 = Sin[11*Pi/32] - Cos[11*Pi/32])*Sqrt[2] = 0.580569354508925 */
  /*   5461/8192 = Cos[11*Pi/32]*Sqrt[2]                  = 0.666655658477747 */
  od_rotate_add_half(r2, r5, r2h, 15679, 14, 1189, 11, 5461, 13, TX_NONE);

  /* 16305/16384 = (Sin[9*Pi/32] + Cos[9*Pi/32])/Sqrt[2] = 0.9951847266721969 */
  /*    803/4096 = (Sin[9*Pi/32] - Cos[9*Pi/32])*Sqrt[2] = 0.1960342806591213 */
  /* 14699/16384 = Cos[9*Pi/32]*Sqrt[2]                  = 0.8971675863426364 */
  od_rotate_sub_half(r4, r3, r4h, 16305, 14, 803, 12, 14699, 14, TX_NONE);

  /* Stage 1 */

  OD_KERNEL_FUNC(od_butterfly_add)(r0, &r0h, r3);
  OD_KERNEL_FUNC(od_butterfly_sub)(r2, &r2h, r1);
  OD_KERNEL_FUNC(od_butterfly_add)(r5, &r5h, r6);
  OD_KERNEL_FUNC(od_butterfly_sub)(r7, &r7h, r4);

  /* Stage 2 */

  OD_KERNEL_FUNC(od_butterfly_add_asym)(r7, r7h, r6);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(r5, r5h, r3);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(r2, r2h, r4);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(r0, r0h, r1);

  /* Stage 3 */

  /*    669/512 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /*  3135/4096 = 2*Cos[3*Pi/8]             = 0.7653668647301796 */
  od_rotate_sub_avg(r3, r4, 669, 9, 8867, 14, 3135, 12, TX_NONE);

  /*    669/512 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /*  3135/4096 = 2*Cos[3*Pi/8]             = 0.7653668647301796 */
  od_rotate_neg_avg(r2, r5, 669, 9, 8867, 14, 3135, 12);

  /*  5793/4096 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951 */
  od_rotate_pi4_sub_avg(r1, r6, 5793, 12, 11585, 13);
}

/**
 * 8-point asymmetric Type-IV iDST
 */
static INLINE void OD_KERNEL_FUNC(od_idst_8_asym)(OD_COEFF *r0, OD_COEFF *r4,
                                                  OD_COEFF *r2, OD_COEFF *r6,
                                                  OD_COEFF *r1, OD_COEFF *r5,
                                                  OD_COEFF *r3, OD_COEFF *r7) {
  OD_COEFF r0h;
  OD_COEFF r2h;
  OD_COEFF r5h;
  OD_COEFF r7h;

  /* Stage 3 */

  /*  5793/4096 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951 */
  od_rotate_pi4_add_avg(r6, r1, 5793, 12, 11585, 13);

  /*    669/512 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /*  3135/4096 = 2*Cos[3*Pi/8]             = 0.7653668647301796 */
  od_rotate_neg_avg(r5, r2, 669, 9, 8867, 14, 3135, 12);

  /*    669/512 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /*  3135/4096 = 2*Cos[3*Pi/8]             = 0.7653668647301796 */
  od_rotate_add_avg(r4, r3, 669, 9, 8867, 14, 3135, 12, TX_NONE);

  /* Stage 2 */

  OD_KERNEL_FUNC(od_butterfly_sub)(r0, &r0h, r1);
  OD_KERNEL_FUNC(od_butterfly_add)(r2, &r2h, r4);
  OD_KERNEL_FUNC(od_butterfly_add)(r5, &r5h, r3);
  OD_KERNEL_FUNC(od_butterfly_add)(r7, &r7h, r6);

  /* Stage 1 */

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(r7, r7h, r4);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(r5, r5h, r6);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(r2, r2h, r1);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(r0, r0h, r3);

  /* Stage 0 */

  /* 16305/16384 = (Sin[9*Pi/32] + Cos[9*Pi/32])/Sqrt[2] = 0.9951847266721969 */
  /*    803/4096 = (Sin[9*Pi/32] - Cos[9*Pi/32])*Sqrt[2] = 0.1960342806591213 */
  /* 14699/16384 = Cos[9*Pi/32]*Sqrt[2]                  = 0.8971675863426364 */
  od_rotate_sub(r4, r3, 16305, 14, 803, 12, 14699, 14, TX_SHIFT);

  /* 15679/16384 = Sin[11*Pi/32] + Cos[11*Pi/32])/Sqrt[2] = 0.956940335732209 */
  /*   1189/2048 = Sin[11*Pi/32] - Cos[11*Pi/32])*Sqrt[2] = 0.580569354508925 */
  /*   5461/8192 = Cos[11*Pi/32]*Sqrt[2]                  = 0.666655658477747 */
  od_rotate_add(r2, r5, 15679, 14, 1189, 11, 5461, 13, TX_SHIFT);

  /* 14449/16384 = Sin[13*Pi/32] + Cos[13*Pi/32])/Sqrt[2] = 0.881921264348355 */
  /* 15447/16384 = Sin[13*Pi/32] - Cos[13*Pi/32])*Sqrt[2] = 0.942793473651995 */
  /*   3363/8192 = Cos[13*Pi/32]*Sqrt[2]                  = 0.410524527522357 */
  od_rotate_sub(r6, r1, 14449, 14, 15447, 14, 3363, 13, TX_SHIFT);

  /* 12665/16384 = (Sin[15*Pi/32] + Cos[15*Pi/32])/Sqrt[2] = 0.77301045336274 */
  /*   5197/4096 = (Sin[15*Pi/32] - Cos[15*Pi/32])*Sqrt[2] = 1.26878656832729 */
  /*  2271/16384 = Cos[15*Pi/32]*Sqrt[2]                   = 0.13861716919909 */
  od_rotate_add(r0, r7, 12665, 14, 5197, 12, 2271, 14, TX_SHIFT);
}

/* --- 16-point Transforms --- */

/**
 * 16-point orthonormal Type-II fDCT
 */
static INLINE void OD_KERNEL_FUNC(od_fdct_16)(OD_COEFF *s0, OD_COEFF *s1,
                                              OD_COEFF *s2, OD_COEFF *s3,
                                              OD_COEFF *s4, OD_COEFF *s5,
                                              OD_COEFF *s6, OD_COEFF *s7,
                                              OD_COEFF *s8, OD_COEFF *s9,
                                              OD_COEFF *sa, OD_COEFF *sb,
                                              OD_COEFF *sc, OD_COEFF *sd,
                                              OD_COEFF *se, OD_COEFF *sf) {
  OD_COEFF s1h;
  OD_COEFF s3h;
  OD_COEFF s5h;
  OD_COEFF s7h;
  OD_COEFF s9h;
  OD_COEFF sbh;
  OD_COEFF sdh;
  OD_COEFF sfh;

  /* +/- Butterflies with asymmetric output. */
  OD_KERNEL_FUNC(od_butterfly_neg)(s0, sf, &sfh);
  OD_KERNEL_FUNC(od_butterfly_add)(s1, &s1h, se);
  OD_KERNEL_FUNC(od_butterfly_neg)(s2, sd, &sdh);
  OD_KERNEL_FUNC(od_butterfly_add)(s3, &s3h, sc);
  OD_KERNEL_FUNC(od_butterfly_neg)(s4, sb, &sbh);
  OD_KERNEL_FUNC(od_butterfly_add)(s5, &s5h, sa);
  OD_KERNEL_FUNC(od_butterfly_neg)(s6, s9, &s9h);
  OD_KERNEL_FUNC(od_butterfly_add)(s7, &s7h, s8);

  /* Embedded 8-point transforms with asymmetric input. */
  OD_KERNEL_FUNC(od_fdct_8_asym)(s0, s1, s1h, s2, s3, s3h, s4, s5, s5h, s6, s7, s7h);
  OD_KERNEL_FUNC(od_fdst_8_asym)(sf, sfh, se, sd, sdh, sc, sb, sbh, sa, s9, s9h, s8);
}

/**
 * 16-point orthonormal Type-II iDCT
 */
static INLINE void OD_KERNEL_FUNC(od_idct_16)(OD_COEFF *s0, OD_COEFF *s8,
                                              OD_COEFF *s4, OD_COEFF *sc,
                                              OD_COEFF *s2, OD_COEFF *sa,
                                              OD_COEFF *s6, OD_COEFF *se,
                                              OD_COEFF *s1, OD_COEFF *s9,
                                              OD_COEFF *s5, OD_COEFF *sd,
                                              OD_COEFF *s3, OD_COEFF *sb,
                                              OD_COEFF *s7, OD_COEFF *sf) {
  OD_COEFF s1h;
  OD_COEFF s3h;
  OD_COEFF s5h;
  OD_COEFF s7h;

  /* Embedded 8-point transforms with asymmetric output. */
  OD_KERNEL_FUNC(od_idst_8_asym)(sf, sb, sd, s9, se, sa, sc, s8);
  OD_KERNEL_FUNC(od_idct_8_asym)(s0, s4, s2, s6, s1, &s1h, s5, &s5h, s3, &s3h, s7, &s7h);

  /* +/- Butterflies with asymmetric input. */
  OD_KERNEL_FUNC(od_butterfly_add_asym)(s7, s7h, s8);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(s6, s9, OD_RSHIFT1(*s9));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(s5, s5h, sa);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(s4, sb, OD_RSHIFT1(*sb));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(s3, s3h, sc);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(s2, sd, OD_RSHIFT1(*sd));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(s1, s1h, se);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(s0, sf, OD_RSHIFT1(*sf));
}

/**
 * 16-point asymmetric Type-II fDCT
 */
static INLINE void OD_KERNEL_FUNC(od_fdct_16_asym)(OD_COEFF *s0, OD_COEFF *s1,
 OD_COEFF s1h, OD_COEFF *s2, OD_COEFF *s3, OD_COEFF s3h, OD_COEFF *s4,
 OD_COEFF *s5, OD_COEFF s5h, OD_COEFF *s6, OD_COEFF *s7, OD_COEFF s7h,
 OD_COEFF *s8, OD_COEFF *s9, OD_COEFF s9h, OD_COEFF *sa, OD_COEFF *sb,
 OD_COEFF sbh, OD_COEFF *sc, OD_COEFF *sd, OD_COEFF sdh, OD_COEFF *se,
 OD_COEFF *sf, OD_COEFF sfh) {

  /* +/- Butterflies with asymmetric input. */
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(s0, sf, sfh);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(s1, s1h, se);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(s2, sd, sdh);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(s3, s3h, sc);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(s4, sb, sbh);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(s5, s5h, sa);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(s6, s9, s9h);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(s7, s7h, s8);

  /* Embedded 8-point orthonormal transforms. */
  OD_KERNEL_FUNC(od_fdct_8)(s0, s1, s2, s3, s4, s5, s6, s7);
  OD_KERNEL_FUNC(od_fdst_8)(sf, se, sd, sc, sb, sa, s9, s8);
}

/**
 * 16-point asymmetric Type-II iDCT
 */
static INLINE void OD_KERNEL_FUNC(od_idct_16_asym)(OD_COEFF *s0, OD_COEFF *s8,
 OD_COEFF *s4, OD_COEFF *sc, OD_COEFF *s2, OD_COEFF *sa, OD_COEFF *s6,
 OD_COEFF *se, OD_COEFF *s1, OD_COEFF *s1h, OD_COEFF *s9, OD_COEFF *s9h,
 OD_COEFF *s5, OD_COEFF *s5h, OD_COEFF *sd, OD_COEFF *sdh, OD_COEFF *s3,
 OD_COEFF *s3h, OD_COEFF *sb, OD_COEFF *sbh, OD_COEFF *s7, OD_COEFF *s7h,
 OD_COEFF *sf, OD_COEFF *sfh) {

  /* Embedded 8-point orthonormal transforms. */
  OD_KERNEL_FUNC(od_idst_8)(sf, sb, sd, s9, se, sa, sc, s8);
  OD_KERNEL_FUNC(od_idct_8)(s0, s4, s2, s6, s1, s5, s3, s7);

  /* +/- Butterflies with asymmetric output. */
  OD_KERNEL_FUNC(od_butterfly_sub)(s7, s7h, s8);
  OD_KERNEL_FUNC(od_butterfly_neg)(s6, s9, s9h);
  OD_KERNEL_FUNC(od_butterfly_sub)(s5, s5h, sa);
  OD_KERNEL_FUNC(od_butterfly_neg)(s4, sb, sbh);
  OD_KERNEL_FUNC(od_butterfly_sub)(s3, s3h, sc);
  OD_KERNEL_FUNC(od_butterfly_neg)(s2, sd, sdh);
  OD_KERNEL_FUNC(od_butterfly_sub)(s1, s1h, se);
  OD_KERNEL_FUNC(od_butterfly_neg)(s0, sf, sfh);
}

/**
 * 16-point orthonormal Type-IV fDST
 */
static INLINE void OD_KERNEL_FUNC(od_fdst_16)(OD_COEFF *s0, OD_COEFF *s1,
                                              OD_COEFF *s2, OD_COEFF *s3,
                                              OD_COEFF *s4, OD_COEFF *s5,
                                              OD_COEFF *s6, OD_COEFF *s7,
                                              OD_COEFF *s8, OD_COEFF *s9,
                                              OD_COEFF *sa, OD_COEFF *sb,
                                              OD_COEFF *sc, OD_COEFF *sd,
                                              OD_COEFF *se, OD_COEFF *sf) {
  OD_COEFF s0h;
  OD_COEFF s2h;
  OD_COEFF sdh;
  OD_COEFF sfh;

  /* Stage 0 */

  /* 24279/32768 = (Sin[31*Pi/64] + Cos[31*Pi/64])/Sqrt[2] = 0.74095112535496 */
  /*  11003/8192 = (Sin[31*Pi/64] - Cos[31*Pi/64])*Sqrt[2] = 1.34311790969404 */
  /*  1137/16384 = Cos[31*Pi/64]*Sqrt[2]                   = 0.06939217050794 */
  od_rotate_add(s0, sf, 24279, 15, 11003, 13, 1137, 14, TX_SHIFT);

  /* 1645/2048 = (Sin[29*Pi/64] + Cos[29*Pi/64])/Sqrt[2] = 0.8032075314806449 */
  /*   305/256 = (Sin[29*Pi/64] - Cos[29*Pi/64])*Sqrt[2] = 1.1913986089848667 */
  /*  425/2048 = Cos[29*Pi/64]*Sqrt[2]                   = 0.2075082269882116 */
  od_rotate_sub(se, s1, 1645, 11, 305, 8, 425, 11, TX_SHIFT);

  /* 14053/32768 = (Sin[27*Pi/64] + Cos[27*Pi/64])/Sqrt[2] = 0.85772861000027 */
  /*   8423/8192 = (Sin[27*Pi/64] - Cos[27*Pi/64])*Sqrt[2] = 1.02820548838644 */
  /*   2815/8192 = Cos[27*Pi/64]*Sqrt[2]                   = 0.34362586580705 */
  od_rotate_add(s2, sd, 14053, 14, 8423, 13, 2815, 13, TX_SHIFT);

  /* 14811/16384 = (Sin[25*Pi/64] + Cos[25*Pi/64])/Sqrt[2] = 0.90398929312344 */
  /*   7005/8192 = (Sin[25*Pi/64] - Cos[25*Pi/64])*Sqrt[2] = 0.85511018686056 */
  /*   3903/8192 = Cos[25*Pi/64]*Sqrt[2]                   = 0.47643419969316 */
  od_rotate_sub(sc, s3, 14811, 14, 7005, 13, 3903, 13, TX_SHIFT);

  /* 30853/32768 = (Sin[23*Pi/64] + Cos[23*Pi/64])/Sqrt[2] = 0.94154406518302 */
  /* 11039/16384 = (Sin[23*Pi/64] - Cos[23*Pi/64])*Sqrt[2] = 0.67377970678444 */
  /*  9907/16384 = Cos[23*Pi/64]*Sqrt[2]                   = 0.60465421179080 */
  od_rotate_add(s4, sb, 30853, 15, 11039, 14, 9907, 14, TX_SHIFT);

  /* 15893/16384 = (Sin[21*Pi/64] + Cos[21*Pi/64])/Sqrt[2] = 0.97003125319454 */
  /*   3981/8192 = (Sin[21*Pi/64] - Cos[21*Pi/64])*Sqrt[2] = 0.89716758634264 */
  /*   1489/2048 = Cos[21*Pi/64]*Sqrt[2]                   = 0.72705107329128 */
  od_rotate_sub(sa, s5, 15893, 14, 3981, 13, 1489, 11, TX_SHIFT);

  /* 32413/32768 = (Sin[19*Pi/64] + Cos[19*Pi/64])/Sqrt[2] = 0.98917650996478 */
  /*    601/2048 = (Sin[19*Pi/64] - Cos[19*Pi/64])*Sqrt[2] = 0.29346094891072 */
  /* 13803/16384 = Cos[19*Pi/64]*Sqrt[2]                   = 0.84244603550942 */
  od_rotate_add(s6, s9, 32413, 15, 601, 11, 13803, 14, TX_SHIFT);

  /* 32729/32768 = (Sin[17*Pi/64] + Cos[17*Pi/64])/Sqrt[2] = 0.99879545620517 */
  /*    201/2048 = (Sin[17*Pi/64] - Cos[17*Pi/64])*Sqrt[2] = 0.09813534865484 */
  /*   1945/2048 = Cos[17*Pi/64]*Sqrt[2]                   = 0.94972778187775 */
  od_rotate_sub(s8, s7, 32729, 15, 201, 11, 1945, 11, TX_SHIFT);

  /* Stage 1 */

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(s0, OD_RSHIFT1(*s0), s7);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(s8, OD_RSHIFT1(*s8), sf);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(s4, OD_RSHIFT1(*s4), s3);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(sc, OD_RSHIFT1(*sc), sb);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(s2, OD_RSHIFT1(*s2), s5);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(sa, OD_RSHIFT1(*sa), sd);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(s6, OD_RSHIFT1(*s6), s1);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(se, OD_RSHIFT1(*se), s9);

  /* Stage 2 */

  OD_KERNEL_FUNC(od_butterfly_add)(s8, NULL, s4);
  OD_KERNEL_FUNC(od_butterfly_add)(s7, NULL, sb);
  OD_KERNEL_FUNC(od_butterfly_sub)(sa, NULL, s6);
  OD_KERNEL_FUNC(od_butterfly_sub)(s5, NULL, s9);
  OD_KERNEL_FUNC(od_butterfly_add)(s0, &s0h, s3);
  OD_KERNEL_FUNC(od_butterfly_add)(sd, &sdh, se);
  OD_KERNEL_FUNC(od_butterfly_sub)(s2, &s2h, s1);
  OD_KERNEL_FUNC(od_butterfly_sub)(sf, &sfh, sc);

  /* Stage 3 */

  /*     301/256 = Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586 */
  /*   1609/2048 = Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022 */
  /* 12785/32768 = 2*Cos[7*Pi/16]              = 0.3901806440322565 */
  od_rotate_add_avg(s8, s7, 301, 8, 1609, 11, 12785, 15, TX_NONE);

  /* 11363/8192 = Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475 */
  /* 9041/32768 = Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431 */
  /*  4551/8192 = Cos[5*Pi/16]                = 0.5555702330196022 */
  od_rotate_add(s9, s6, 11363, 13, 9041, 15, 4551, 13, TX_NONE);

  /*  5681/4096 = Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475 */
  /* 9041/32768 = Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431 */
  /*  4551/4096 = 2*Cos[5*Pi/16]              = 1.1111404660392044 */
  od_rotate_neg_avg(s5, sa, 5681, 12, 9041, 15, 4551, 12);

  /*   9633/8192 = Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586 */
  /* 12873/16384 = Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022 */
  /*  6393/32768 = Cos[7*Pi/16]                = 0.1950903220161283 */
  od_rotate_neg(s4, sb, 9633, 13, 12873, 14, 6393, 15);

  /* Stage 4 */

  OD_KERNEL_FUNC(od_butterfly_add_asym)(s2, s2h, sc);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(s0, s0h, s1);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(sf, sfh, se);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(sd, sdh, s3);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(s7, OD_RSHIFT1(*s7), s6);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(s8, OD_RSHIFT1(*s8), s9);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(sa, OD_RSHIFT1(*sa), sb);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(s5, OD_RSHIFT1(*s5), s4);

  /* Stage 5 */

  /*    669/512 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /*  3135/4096 = 2*Cos[7*Pi/8]             = 0.7653668647301796 */
  od_rotate_add_avg(sc, s3, 669, 9, 8867, 14, 3135, 12, TX_NONE);

  /*    669/512 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3870398453221475 */
  /* 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /*  3135/4096 = 2*Cos[3*Pi/8]             = 0.7653668647301796 */
  od_rotate_neg_avg(s2, sd, 669, 9, 8867, 14, 3135, 12);

  /*  5793/4096 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951 */
  od_rotate_pi4_add_avg(sa, s5, 5793, 12, 11585, 13);

  /*  5793/4096 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951 */
  od_rotate_pi4_add_avg(s6, s9, 5793, 12, 11585, 13);

  /*  5793/4096 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951 */
  od_rotate_pi4_add_avg(se, s1, 5793, 12, 11585, 13);
}

/**
 * 16-point orthonormal Type-IV iDST
 */
static INLINE void OD_KERNEL_FUNC(od_idst_16)(OD_COEFF *s0, OD_COEFF *s8,
                                              OD_COEFF *s4, OD_COEFF *sc,
                                              OD_COEFF *s2, OD_COEFF *sa,
                                              OD_COEFF *s6, OD_COEFF *se,
                                              OD_COEFF *s1, OD_COEFF *s9,
                                              OD_COEFF *s5, OD_COEFF *sd,
                                              OD_COEFF *s3, OD_COEFF *sb,
                                              OD_COEFF *s7, OD_COEFF *sf) {
  OD_COEFF s0h;
  OD_COEFF s2h;
  OD_COEFF s4h;
  OD_COEFF s6h;
  OD_COEFF s8h;
  OD_COEFF sah;
  OD_COEFF sch;
  OD_COEFF sdh;
  OD_COEFF seh;
  OD_COEFF sfh;

  /* Stage 5 */

  /*  5793/4096 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951 */
  od_rotate_pi4_add_avg(s6, s9, 5793, 12, 11585, 13);

  /*  5793/4096 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951 */
  od_rotate_pi4_add_avg(sa, s5, 5793, 12, 11585, 13);

  /*  5793/4096 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951 */
  od_rotate_pi4_add_avg(se, s1, 5793, 12, 11585, 13);

  /*     669/512 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /*  8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /*   3135/4096 = 2*Cos[7*Pi/8]             = 0.7653668647301796 */
  od_rotate_add_avg(sc, s3, 669, 9, 8867, 14, 3135, 12, TX_NONE);

  /*    669/512 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3870398453221475 */
  /* 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /*  3135/4096 = 2*Cos[3*Pi/8]             = 0.7653668647301796 */
  od_rotate_neg_avg(sd, s2, 669, 9, 8867, 14, 3135, 12);

  /* Stage 4 */

  OD_KERNEL_FUNC(od_butterfly_add)(s5, NULL, s4);
  OD_KERNEL_FUNC(od_butterfly_sub)(sa, NULL, sb);
  OD_KERNEL_FUNC(od_butterfly_sub)(s8, NULL, s9);
  OD_KERNEL_FUNC(od_butterfly_add)(s7, NULL, s6);
  OD_KERNEL_FUNC(od_butterfly_add)(sd, &sdh, s3);
  OD_KERNEL_FUNC(od_butterfly_add)(sf, &sfh, se);
  OD_KERNEL_FUNC(od_butterfly_sub)(s0, &s0h, s1);
  OD_KERNEL_FUNC(od_butterfly_add)(s2, &s2h, sc);

  /* Stage 3 */

  /*     301/256 = Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586 */
  /*   1609/2048 = Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022 */
  /* 12785/32768 = 2*Cos[7*Pi/16]              = 0.3901806440322565 */
  od_rotate_add_avg(s8, s7, 301, 8, 1609, 11, 12785, 15, TX_NONE);

  /* 11363/8192 = Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475 */
  /* 9041/32768 = Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431 */
  /*  4551/8192 = Cos[5*Pi/16]                = 0.5555702330196022 */
  od_rotate_add(s9, s6, 11363, 13, 9041, 15, 4551, 13, TX_NONE);

  /*  5681/4096 = Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475 */
  /* 9041/32768 = Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431 */
  /*  4551/4096 = 2*Cos[5*Pi/16]              = 1.1111404660392044 */
  od_rotate_neg_avg(sa, s5, 5681, 12, 9041, 15, 4551, 12);

  /*   9633/8192 = Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586 */
  /* 12873/16384 = Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022 */
  /*  6393/32768 = Cos[7*Pi/16]                = 0.1950903220161283 */
  od_rotate_neg(sb, s4, 9633, 13, 12873, 14, 6393, 15);

  /* Stage 2 */

  OD_KERNEL_FUNC(od_butterfly_add_asym)(s8, OD_RSHIFT1(*s8), s4);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(s7, OD_RSHIFT1(*s7), sb);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(sa, OD_RSHIFT1(*sa), s6);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(s5, OD_RSHIFT1(*s5), s9);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(s0, s0h, s3);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(sd, sdh, se);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(s2, s2h, s1);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(sf, sfh, sc);

  /* Stage 1 */

  OD_KERNEL_FUNC(od_butterfly_sub)(s0, &s0h, s7);
  OD_KERNEL_FUNC(od_butterfly_sub)(s8, &s8h, sf);
  OD_KERNEL_FUNC(od_butterfly_add)(s4, &s4h, s3);
  OD_KERNEL_FUNC(od_butterfly_add)(sc, &sch, sb);
  OD_KERNEL_FUNC(od_butterfly_sub)(s2, &s2h, s5);
  OD_KERNEL_FUNC(od_butterfly_sub)(sa, &sah, sd);
  OD_KERNEL_FUNC(od_butterfly_add)(s6, &s6h, s1);
  OD_KERNEL_FUNC(od_butterfly_add)(se, &seh, s9);

  /* Stage 0 */

  /*   4091/4096 = (Sin[17*Pi/64] + Cos[17*Pi/64])/Sqrt[2] = 0.99879545620517 */
  /*    201/2048 = (Sin[17*Pi/64] - Cos[17*Pi/64])*Sqrt[2] = 0.09813534865484 */
  /* 31121/32768 = Cos[17*Pi/64]*Sqrt[2]                   = 0.94972778187775 */
  od_rotate_sub_half(s8, s7, s8h, 4091, 12, 201, 11, 31121, 15, TX_NONE);

  /* 16207/16384 = (Sin[19*Pi/64] + Cos[19*Pi/64])/Sqrt[2] = 0.98917650996478 */
  /*    601/2048 = (Sin[19*Pi/64] - Cos[19*Pi/64])*Sqrt[2] = 0.29346094891072 */
  /* 27605/32768 = Cos[19*Pi/64]*Sqrt[2]                   = 0.84244603550942 */
  od_rotate_add_half(s6, s9, s6h, 16207, 14, 601, 11, 27605, 15, TX_NONE);

  /* 15893/16384 = (Sin[21*Pi/64] + Cos[21*Pi/64])/Sqrt[2] = 0.97003125319454 */
  /*   3981/8192 = (Sin[21*Pi/64] - Cos[21*Pi/64])*Sqrt[2] = 0.89716758634264 */
  /*   1489/2048 = Cos[21*Pi/64]*Sqrt[2]                   = 0.72705107329128 */
  od_rotate_sub_half(sa, s5, sah, 15893, 14, 3981, 13, 1489, 11, TX_NONE);

  /*   7713/8192 = (Sin[23*Pi/64] + Cos[23*Pi/64])/Sqrt[2] = 0.94154406518302 */
  /* 11039/16384 = (Sin[23*Pi/64] - Cos[23*Pi/64])*Sqrt[2] = 0.67377970678444 */
  /* 19813/32768 = Cos[23*Pi/64]*Sqrt[2]                   = 0.60465421179080 */
  od_rotate_add_half(s4, sb, s4h, 7713, 13, 11039, 14, 19813, 15, TX_NONE);

  /* 14811/16384 = (Sin[25*Pi/64] + Cos[25*Pi/64])/Sqrt[2] = 0.90398929312344 */
  /*   7005/8192 = (Sin[25*Pi/64] - Cos[25*Pi/64])*Sqrt[2] = 0.85511018686056 */
  /*   3903/8192 = Cos[25*Pi/64]*Sqrt[2]                   = 0.47643419969316 */
  od_rotate_sub_half(sc, s3, sch, 14811, 14, 7005, 13, 3903, 13, TX_NONE);

  /* 14053/32768 = (Sin[27*Pi/64] + Cos[27*Pi/64])/Sqrt[2] = 0.85772861000027 */
  /*   8423/8192 = (Sin[27*Pi/64] - Cos[27*Pi/64])*Sqrt[2] = 1.02820548838644 */
  /*   2815/8192 = Cos[27*Pi/64]*Sqrt[2]                   = 0.34362586580705 */
  od_rotate_add_half(s2, sd, s2h, 14053, 14, 8423, 13, 2815, 13, TX_NONE);

  /* 1645/2048 = (Sin[29*Pi/64] + Cos[29*Pi/64])/Sqrt[2] = 0.8032075314806449 */
  /*   305/256 = (Sin[29*Pi/64] - Cos[29*Pi/64])*Sqrt[2] = 1.1913986089848667 */
  /*  425/2048 = Cos[29*Pi/64]*Sqrt[2]                   = 0.2075082269882116 */
  od_rotate_sub_half(se, s1, seh, 1645, 11, 305, 8, 425, 11, TX_NONE);

  /*  3035/32768 = (Sin[31*Pi/64] + Cos[31*Pi/64])/Sqrt[2] = 0.74095112535496 */
  /* 44011/32768 = (Sin[31*Pi/64] - Cos[31*Pi/64])*Sqrt[2] = 1.34311790969404 */
  /*  1137/16384 = Cos[31*Pi/64]*Sqrt[2]                   = 0.06939217050794 */
  od_rotate_add_half(s0, sf, s0h, 3035, 12, 44011, 15, 1137, 14, TX_NONE);
}

static INLINE void OD_KERNEL_FUNC(od_flip_idst_16)(OD_COEFF *s0, OD_COEFF *s8,
 OD_COEFF *s4, OD_COEFF *sc, OD_COEFF *s2, OD_COEFF *sa, OD_COEFF *s6,
 OD_COEFF *se, OD_COEFF *s1, OD_COEFF *s9, OD_COEFF *s5, OD_COEFF *sd,
 OD_COEFF *s3, OD_COEFF *sb, OD_COEFF *s7, OD_COEFF *sf) {
  OD_KERNEL_FUNC(od_idst_16)(
   s0, s8, s4, sc, s2, sa, s6, se, s1, s9, s5, sd, s3, sb, s7, sf);
  OD_SWAP(s0, sf);
  OD_SWAP(s8, s7);
  OD_SWAP(s4, sb);
  OD_SWAP(sc, s3);
  OD_SWAP(s2, sd);
  OD_SWAP(sa, s5);
  OD_SWAP(s6, s9);
  OD_SWAP(se, s1);
}

/**
 * 16-point asymmetric Type-IV fDST
 */
static INLINE void OD_KERNEL_FUNC(od_fdst_16_asym)(OD_COEFF *s0, OD_COEFF s0h,
 OD_COEFF *s1, OD_COEFF *s2, OD_COEFF s2h, OD_COEFF *s3, OD_COEFF *s4,
 OD_COEFF s4h, OD_COEFF *s5, OD_COEFF *s6, OD_COEFF s6h, OD_COEFF *s7,
 OD_COEFF *s8, OD_COEFF s8h, OD_COEFF *s9, OD_COEFF *sa, OD_COEFF sah,
 OD_COEFF *sb, OD_COEFF *sc, OD_COEFF sch, OD_COEFF *sd, OD_COEFF *se,
 OD_COEFF seh, OD_COEFF *sf) {
  OD_COEFF sdh;
  OD_COEFF sfh;

  /* Stage 0 */

  /*   1073/2048 = (Sin[31*Pi/64] + Cos[31*Pi/64])/2 = 0.5239315652662953 */
  /* 62241/32768 = (Sin[31*Pi/64] - Cos[31*Pi/64])*2 = 1.8994555637555088 */
  /*   201/16384 = Cos[31*Pi/64]*2                   = 0.0981353486548360 */
  od_rotate_add_half(s0, sf, s0h, 1073, 11, 62241, 15, 201, 11, TX_SHIFT);

  /* 18611/32768 = (Sin[29*Pi/64] + Cos[29*Pi/64])/2 = 0.5679534922100714 */
  /* 55211/32768 = (Sin[29*Pi/64] - Cos[29*Pi/64])*2 = 1.6848920710188384 */
  /*    601/2048 = Cos[29*Pi/64]*2                   = 0.2934609489107235 */
  od_rotate_sub_half(se, s1, seh, 18611, 15, 55211, 15, 601, 11, TX_SHIFT);

  /*  9937/16384 = (Sin[27*Pi/64] + Cos[27*Pi/64])/2 = 0.6065057165489039 */
  /*   1489/1024 = (Sin[27*Pi/64] - Cos[27*Pi/64])*2 = 1.4541021465825602 */
  /*   3981/8192 = Cos[27*Pi/64]*2                   = 0.4859603598065277 */
  od_rotate_add_half(s2, sd, s2h, 9937, 14, 1489, 10, 3981, 13, TX_SHIFT);

  /* 10473/16384 = (Sin[25*Pi/64] + Cos[25*Pi/64])/2 = 0.6392169592876205 */
  /* 39627/32768 = (Sin[25*Pi/64] - Cos[25*Pi/64])*2 = 1.2093084235816014 */
  /* 11039/16384 = Cos[25*Pi/64]*2                   = 0.6737797067844401 */
  od_rotate_sub_half(sc, s3, sch, 10473, 14, 39627, 15, 11039, 14, TX_SHIFT);

  /* 2727/4096 = (Sin[23*Pi/64] + Cos[23*Pi/64])/2 = 0.6657721932768628 */
  /* 3903/4096 = (Sin[23*Pi/64] - Cos[23*Pi/64])*2 = 0.9528683993863225 */
  /* 7005/8192 = Cos[23*Pi/64]*2                   = 0.8551101868605642 */
  od_rotate_add_half(s4, sb, s4h, 2727, 12, 3903, 12, 7005, 13, TX_SHIFT);

  /* 5619/8192 = (Sin[21*Pi/64] + Cos[21*Pi/64])/2 = 0.6859156770967569 */
  /* 2815/4096 = (Sin[21*Pi/64] - Cos[21*Pi/64])*2 = 0.6872517316141069 */
  /* 8423/8192 = Cos[21*Pi/64]*2                   = 1.0282054883864433 */
  od_rotate_sub_half(sa, s5, sah, 5619, 13, 2815, 12, 8423, 13, TX_SHIFT);

  /*   2865/4096 = (Sin[19*Pi/64] + Cos[19*Pi/64])/2 = 0.6994534179865391 */
  /* 13588/32768 = (Sin[19*Pi/64] - Cos[19*Pi/64])*2 = 0.4150164539764232 */
  /*     305/256 = Cos[19*Pi/64]*2                   = 1.1913986089848667 */
  od_rotate_add_half(s6, s9, s6h, 2865, 12, 13599, 15, 305, 8, TX_SHIFT);

  /* 23143/32768 = (Sin[17*Pi/64] + Cos[17*Pi/64])/2 = 0.7062550401009887 */
  /*   1137/8192 = (Sin[17*Pi/64] - Cos[17*Pi/64])*2 = 0.1387843410158816 */
  /*  11003/8192 = Cos[17*Pi/64]*2                   = 1.3431179096940367 */
  od_rotate_sub_half(s8, s7, s8h, 23143, 15, 1137, 13, 11003, 13, TX_SHIFT);

  /* Stage 1 */

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(s0, OD_RSHIFT1(*s0), s7);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(s8, OD_RSHIFT1(*s8), sf);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(s4, OD_RSHIFT1(*s4), s3);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(sc, OD_RSHIFT1(*sc), sb);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(s2, OD_RSHIFT1(*s2), s5);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(sa, OD_RSHIFT1(*sa), sd);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(s6, OD_RSHIFT1(*s6), s1);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(se, OD_RSHIFT1(*se), s9);

  /* Stage 2 */

  OD_KERNEL_FUNC(od_butterfly_add)(s8, NULL, s4);
  OD_KERNEL_FUNC(od_butterfly_add)(s7, NULL, sb);
  OD_KERNEL_FUNC(od_butterfly_sub)(sa, NULL, s6);
  OD_KERNEL_FUNC(od_butterfly_sub)(s5, NULL, s9);
  OD_KERNEL_FUNC(od_butterfly_add)(s0, &s0h, s3);
  OD_KERNEL_FUNC(od_butterfly_add)(sd, &sdh, se);
  OD_KERNEL_FUNC(od_butterfly_sub)(s2, &s2h, s1);
  OD_KERNEL_FUNC(od_butterfly_sub)(sf, &sfh, sc);

  /* Stage 3 */

  /*   9633/8192 = Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586 */
  /* 12873/16384 = Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022 */
  /*  6393/32768 = Cos[7*Pi/16]                = 0.1950903220161283 */
  od_rotate_add(s8, s7, 9633, 13, 12873, 14, 6393, 15, TX_NONE);

  /* 45451/32768 = Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475 */
  /*  9041/32768 = Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431 */
  /*   4551/8192 = Cos[5*Pi/16]                = 0.5555702330196022 */
  od_rotate_add(s9, s6, 22725, 14, 9041, 15, 4551, 13, TX_NONE);

  /*  11363/8192 = Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475 */
  /*  9041/32768 = Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431 */
  /*   4551/8192 = Cos[5*Pi/16]                = 0.5555702330196022 */
  od_rotate_neg(s5, sa, 11363, 13, 9041, 15, 4551, 13);

  /*  9633/32768 = Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586 */
  /* 12873/16384 = Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022 */
  /*  6393/32768 = Cos[7*Pi/16]                = 0.1950903220161283 */
  od_rotate_neg(s4, sb, 9633, 13, 12873, 14, 6393, 15);

  /* Stage 4 */

  OD_KERNEL_FUNC(od_butterfly_add_asym)(s2, s2h, sc);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(s0, s0h, s1);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(sf, sfh, se);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(sd, sdh, s3);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(s7, OD_RSHIFT1(*s7), s6);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(s8, OD_RSHIFT1(*s8), s9);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(sa, OD_RSHIFT1(*sa), sb);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(s5, OD_RSHIFT1(*s5), s4);

  /* Stage 5 */

  /*  10703/8192 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /*  8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /*   3135/8192 = Cos[7*Pi/8]               = 0.3826834323650898 */
  od_rotate_add(sc, s3, 10703, 13, 8867, 14, 3135, 13, TX_NONE);

  /*  10703/8192 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3870398453221475 */
  /*  8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /*   3135/8192 = Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_neg(s2, sd, 10703, 13, 8867, 14, 3135, 13);

  /* 11585/8192 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /*  5793/8192 = Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(sa, s5, 11585, 13, 5793, 13);

  /* 11585/8192 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /*  5793/8192 = Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(s6, s9, 11585, 13, 5793, 13);

  /* 11585/8192 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /*  5793/8192 = Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(se, s1, 11585, 13, 5793, 13);
}

/**
 * 16-point asymmetric Type-IV iDST
 */
static INLINE void OD_KERNEL_FUNC(od_idst_16_asym)(OD_COEFF *s0, OD_COEFF *s8,
 OD_COEFF *s4, OD_COEFF *sc, OD_COEFF *s2, OD_COEFF *sa, OD_COEFF *s6,
 OD_COEFF *se, OD_COEFF *s1, OD_COEFF *s9, OD_COEFF *s5, OD_COEFF *sd,
 OD_COEFF *s3, OD_COEFF *sb, OD_COEFF *s7, OD_COEFF *sf) {
  OD_COEFF s0h;
  OD_COEFF s2h;
  OD_COEFF s4h;
  OD_COEFF s6h;
  OD_COEFF s8h;
  OD_COEFF sah;
  OD_COEFF sch;
  OD_COEFF sdh;
  OD_COEFF seh;
  OD_COEFF sfh;

  /* Stage 5 */

  /* 11585/8192 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /*  5793/8192 = Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(s6, s9, 11585, 13, 5793, 13);

  /* 11585/8192 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /*  5793/8192 = Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(sa, s5, 11585, 13, 5793, 13);

  /* 11585/8192 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /*  5793/8192 = Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(se, s1, 11585, 13, 5793, 13);

  /*  10703/8192 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /*  8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /*   3135/8192 = Cos[7*Pi/8]               = 0.7653668647301796 */
  od_rotate_add(sc, s3, 10703, 13, 8867, 14, 3135, 13, TX_NONE);

  /*  10703/8192 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3870398453221475 */
  /*  8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /*   3135/8192 = Cos[3*Pi/8]               = 0.7653668647301796 */
  od_rotate_neg(sd, s2, 10703, 13, 8867, 14, 3135, 13);

  /* Stage 4 */

  OD_KERNEL_FUNC(od_butterfly_add)(s5, NULL, s4);
  OD_KERNEL_FUNC(od_butterfly_sub)(sa, NULL, sb);
  OD_KERNEL_FUNC(od_butterfly_sub)(s8, NULL, s9);
  OD_KERNEL_FUNC(od_butterfly_add)(s7, NULL, s6);
  OD_KERNEL_FUNC(od_butterfly_add)(sd, &sdh, s3);
  OD_KERNEL_FUNC(od_butterfly_add)(sf, &sfh, se);
  OD_KERNEL_FUNC(od_butterfly_sub)(s0, &s0h, s1);
  OD_KERNEL_FUNC(od_butterfly_add)(s2, &s2h, sc);

  /* Stage 3 */

  /*   9633/8192 = Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586 */
  /* 12873/16384 = Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022 */
  /*  6393/32768 = Cos[7*Pi/16]                = 0.1950903220161283 */
  od_rotate_neg(sb, s4, 9633, 13, 12873, 14, 6393, 15);

  /*  11363/8192 = Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475 */
  /*  9041/32768 = Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431 */
  /*   4551/8192 = Cos[5*Pi/16]                = 0.5555702330196022 */
  od_rotate_neg(sa, s5, 11363, 13, 9041, 15, 4551, 13);

  /* 22725/16384 = Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475 */
  /*  9041/32768 = Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431 */
  /* 18205/32768 = Cos[5*Pi/16]                = 0.5555702330196022 */
  od_rotate_add(s9, s6, 22725, 14, 9041, 15, 18205, 15, TX_NONE);

  /*  9633/8192 = Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586 */
  /*  1609/2048 = Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022 */
  /* 6393/32768 = Cos[7*Pi/16]                = 0.1950903220161283 */
  od_rotate_add(s8, s7, 9633, 13, 1609, 11, 6393, 15, TX_NONE);

  /* Stage 2 */

  OD_KERNEL_FUNC(od_butterfly_add_asym)(s8, OD_RSHIFT1(*s8), s4);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(s7, OD_RSHIFT1(*s7), sb);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(sa, OD_RSHIFT1(*sa), s6);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(s5, OD_RSHIFT1(*s5), s9);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(s0, s0h, s3);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(sd, sdh, se);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(s2, s2h, s1);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(sf, sfh, sc);

  /* Stage 1 */

  OD_KERNEL_FUNC(od_butterfly_sub)(s0, &s0h, s7);
  OD_KERNEL_FUNC(od_butterfly_sub)(s8, &s8h, sf);
  OD_KERNEL_FUNC(od_butterfly_add)(s4, &s4h, s3);
  OD_KERNEL_FUNC(od_butterfly_add)(sc, &sch, sb);
  OD_KERNEL_FUNC(od_butterfly_sub)(s2, &s2h, s5);
  OD_KERNEL_FUNC(od_butterfly_sub)(sa, &sah, sd);
  OD_KERNEL_FUNC(od_butterfly_add)(s6, &s6h, s1);
  OD_KERNEL_FUNC(od_butterfly_add)(se, &seh, s9);

  /* Stage 0 */

  /* 23143/32768 = (Sin[17*Pi/64] + Cos[17*Pi/64])/2 = 0.7062550401009887 */
  /*   1137/8192 = (Sin[17*Pi/64] - Cos[17*Pi/64])*2 = 0.1387843410158816 */
  /* 44011/32768 = Cos[17*Pi/64]*2                   = 1.3431179096940367 */
  od_rotate_sub_half(s8, s7, s8h, 23143, 15, 1137, 13, 44011, 15, TX_SHIFT);

  /*   2865/4096 = (Sin[19*Pi/64] + Cos[19*Pi/64])/2 = 0.6994534179865391 */
  /* 13599/32768 = (Sin[19*Pi/64] - Cos[19*Pi/64])*2 = 0.4150164539764232 */
  /*     305/256 = Cos[19*Pi/64]*2                   = 1.1913986089848667 */
  od_rotate_add_half(s6, s9, s6h, 2865, 12, 13599, 15, 305, 8, TX_SHIFT);

  /* 5619/8192 = (Sin[21*Pi/64] + Cos[21*Pi/64])/2 = 0.6859156770967569 */
  /* 2815/4096 = (Sin[21*Pi/64] - Cos[21*Pi/64])*2 = 0.6872517316141069 */
  /* 8423/8192 = Cos[21*Pi/64]*2                   = 1.0282054883864433 */
  od_rotate_sub_half(sa, s5, sah, 5619, 13, 2815, 12, 8423, 13, TX_SHIFT);

  /* 2727/4096 = (Sin[23*Pi/64] + Cos[23*Pi/64])/2 = 0.6657721932768628 */
  /* 3903/4096 = (Sin[23*Pi/64] - Cos[23*Pi/64])*2 = 0.9528683993863225 */
  /* 7005/8192 = Cos[23*Pi/64]*2                   = 0.8551101868605642 */
  od_rotate_add_half(s4, sb, s4h, 2727, 12, 3903, 12, 7005, 13, TX_SHIFT);

  /* 10473/16384 = (Sin[25*Pi/64] + Cos[25*Pi/64])/2 = 0.6392169592876205 */
  /* 39627/32768 = (Sin[25*Pi/64] - Cos[25*Pi/64])*2 = 1.2093084235816014 */
  /* 11039/16384 = Cos[25*Pi/64]*2                   = 0.6737797067844401 */
  od_rotate_sub_half(sc, s3, sch, 10473, 14, 39627, 15, 11039, 14, TX_SHIFT);

  /*  9937/16384 = (Sin[27*Pi/64] + Cos[27*Pi/64])/2 = 0.6065057165489039 */
  /*   1489/1024 = (Sin[27*Pi/64] - Cos[27*Pi/64])*2 = 1.4541021465825602 */
  /*   3981/8192 = Cos[27*Pi/64]*2                   = 0.4859603598065277 */
  od_rotate_add_half(s2, sd, s2h, 9937, 14, 1489, 10, 3981, 13, TX_SHIFT);

  /* 18611/32768 = (Sin[29*Pi/64] + Cos[29*Pi/64])/2 = 0.5679534922100714 */
  /* 55211/32768 = (Sin[29*Pi/64] - Cos[29*Pi/64])*2 = 1.6848920710188384 */
  /*    601/2048 = Cos[29*Pi/64]*2                   = 0.2934609489107235 */
  od_rotate_sub_half(se, s1, seh, 18611, 15, 55211, 15, 601, 11, TX_SHIFT);

  /*   1073/2048 = (Sin[31*Pi/64] + Cos[31*Pi/64])/2 = 0.5239315652662953 */
  /* 62241/32768 = (Sin[31*Pi/64] - Cos[31*Pi/64])*2 = 1.8994555637555088 */
  /*    201/2048 = Cos[31*Pi/64]*2                   = 0.0981353486548360 */
  od_rotate_add_half(s0, sf, s0h, 1073, 11, 62241, 15, 201, 11, TX_SHIFT);
}

/* --- 32-point Transforms --- */

/**
 * 32-point orthonormal Type-II fDCT
 */
static INLINE void OD_KERNEL_FUNC(od_fdct_32)(OD_COEFF *t0, OD_COEFF *t1,
                                              OD_COEFF *t2, OD_COEFF *t3,
                                              OD_COEFF *t4, OD_COEFF *t5,
                                              OD_COEFF *t6, OD_COEFF *t7,
                                              OD_COEFF *t8, OD_COEFF *t9,
                                              OD_COEFF *ta, OD_COEFF *tb,
                                              OD_COEFF *tc, OD_COEFF *td,
                                              OD_COEFF *te, OD_COEFF *tf,
                                              OD_COEFF *tg, OD_COEFF *th,
                                              OD_COEFF *ti, OD_COEFF *tj,
                                              OD_COEFF *tk, OD_COEFF *tl,
                                              OD_COEFF *tm, OD_COEFF *tn,
                                              OD_COEFF *to, OD_COEFF *tp,
                                              OD_COEFF *tq, OD_COEFF *tr,
                                              OD_COEFF *ts, OD_COEFF *tt,
                                              OD_COEFF *tu, OD_COEFF *tv) {
  OD_COEFF t1h;
  OD_COEFF t3h;
  OD_COEFF t5h;
  OD_COEFF t7h;
  OD_COEFF t9h;
  OD_COEFF tbh;
  OD_COEFF tdh;
  OD_COEFF tfh;
  OD_COEFF thh;
  OD_COEFF tjh;
  OD_COEFF tlh;
  OD_COEFF tnh;
  OD_COEFF tph;
  OD_COEFF trh;
  OD_COEFF tth;
  OD_COEFF tvh;

  /* +/- Butterflies with asymmetric output. */
  OD_KERNEL_FUNC(od_butterfly_neg)(t0, tv, &tvh);
  OD_KERNEL_FUNC(od_butterfly_add)(t1, &t1h, tu);
  OD_KERNEL_FUNC(od_butterfly_neg)(t2, tt, &tth);
  OD_KERNEL_FUNC(od_butterfly_add)(t3, &t3h, ts);
  OD_KERNEL_FUNC(od_butterfly_neg)(t4, tr, &trh);
  OD_KERNEL_FUNC(od_butterfly_add)(t5, &t5h, tq);
  OD_KERNEL_FUNC(od_butterfly_neg)(t6, tp, &tph);
  OD_KERNEL_FUNC(od_butterfly_add)(t7, &t7h, to);
  OD_KERNEL_FUNC(od_butterfly_neg)(t8, tn, &tnh);
  OD_KERNEL_FUNC(od_butterfly_add)(t9, &t9h, tm);
  OD_KERNEL_FUNC(od_butterfly_neg)(ta, tl, &tlh);
  OD_KERNEL_FUNC(od_butterfly_add)(tb, &tbh, tk);
  OD_KERNEL_FUNC(od_butterfly_neg)(tc, tj, &tjh);
  OD_KERNEL_FUNC(od_butterfly_add)(td, &tdh, ti);
  OD_KERNEL_FUNC(od_butterfly_neg)(te, th, &thh);
  OD_KERNEL_FUNC(od_butterfly_add)(tf, &tfh, tg);

  /* Embedded 16-point transforms with asymmetric input. */
  OD_KERNEL_FUNC(od_fdct_16_asym)(
   t0, t1, t1h, t2, t3, t3h, t4, t5, t5h, t6, t7, t7h,
   t8, t9, t9h, ta, tb, tbh, tc, td, tdh, te, tf, tfh);
  OD_KERNEL_FUNC(od_fdst_16_asym)(
   tv, tvh, tu, tt, tth, ts, tr, trh, tq, tp, tph, to,
   tn, tnh, tm, tl, tlh, tk, tj, tjh, ti, th, thh, tg);
}

/**
 * 32-point orthonormal Type-II iDCT
 */
static INLINE void OD_KERNEL_FUNC(od_idct_32)(OD_COEFF *t0, OD_COEFF *tg,
                                              OD_COEFF *t8, OD_COEFF *to,
                                              OD_COEFF *t4, OD_COEFF *tk,
                                              OD_COEFF *tc, OD_COEFF *ts,
                                              OD_COEFF *t2, OD_COEFF *ti,
                                              OD_COEFF *ta, OD_COEFF *tq,
                                              OD_COEFF *t6, OD_COEFF *tm,
                                              OD_COEFF *te, OD_COEFF *tu,
                                              OD_COEFF *t1, OD_COEFF *th,
                                              OD_COEFF *t9, OD_COEFF *tp,
                                              OD_COEFF *t5, OD_COEFF *tl,
                                              OD_COEFF *td, OD_COEFF *tt,
                                              OD_COEFF *t3, OD_COEFF *tj,
                                              OD_COEFF *tb, OD_COEFF *tr,
                                              OD_COEFF *t7, OD_COEFF *tn,
                                              OD_COEFF *tf, OD_COEFF *tv) {
  OD_COEFF t1h;
  OD_COEFF t3h;
  OD_COEFF t5h;
  OD_COEFF t7h;
  OD_COEFF t9h;
  OD_COEFF tbh;
  OD_COEFF tdh;
  OD_COEFF tfh;

  /* Embedded 16-point transforms with asymmetric output. */
  OD_KERNEL_FUNC(od_idst_16_asym)(
   tv, tn, tr, tj, tt, tl, tp, th, tu, tm, tq, ti, ts, tk, to, tg);
  OD_KERNEL_FUNC(od_idct_16_asym)(
   t0, t8, t4, tc, t2, ta, t6, te,
   t1, &t1h, t9, &t9h, t5, &t5h, td, &tdh,
   t3, &t3h, tb, &tbh, t7, &t7h, tf, &tfh);

  /* +/- Butterflies with asymmetric input. */
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tf, tfh, tg);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(te, th, OD_RSHIFT1(*th));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(td, tdh, ti);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(tc, tj, OD_RSHIFT1(*tj));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tb, tbh, tk);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(ta, tl, OD_RSHIFT1(*tl));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t9, t9h, tm);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(t8, tn, OD_RSHIFT1(*tn));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t7, t7h, to);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(t6, tp, OD_RSHIFT1(*tp));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t5, t5h, tq);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(t4, tr, OD_RSHIFT1(*tr));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t3, t3h, ts);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(t2, tt, OD_RSHIFT1(*tt));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t1, t1h, tu);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(t0, tv, OD_RSHIFT1(*tv));
}

/**
 * 32-point asymmetric Type-II fDCT
 */
static INLINE void OD_KERNEL_FUNC(od_fdct_32_asym)(OD_COEFF *t0, OD_COEFF *t1,
 OD_COEFF t1h, OD_COEFF *t2, OD_COEFF *t3, OD_COEFF t3h, OD_COEFF *t4,
 OD_COEFF *t5, OD_COEFF t5h, OD_COEFF *t6, OD_COEFF *t7, OD_COEFF t7h,
 OD_COEFF *t8, OD_COEFF *t9, OD_COEFF t9h, OD_COEFF *ta, OD_COEFF *tb,
 OD_COEFF tbh, OD_COEFF *tc, OD_COEFF *td, OD_COEFF tdh, OD_COEFF *te,
 OD_COEFF *tf, OD_COEFF tfh, OD_COEFF *tg, OD_COEFF *th, OD_COEFF thh,
 OD_COEFF *ti, OD_COEFF *tj, OD_COEFF tjh, OD_COEFF *tk, OD_COEFF *tl,
 OD_COEFF tlh, OD_COEFF *tm, OD_COEFF *tn, OD_COEFF tnh, OD_COEFF *to,
 OD_COEFF *tp, OD_COEFF tph, OD_COEFF *tq, OD_COEFF *tr, OD_COEFF trh,
 OD_COEFF *ts, OD_COEFF *tt, OD_COEFF tth, OD_COEFF *tu, OD_COEFF *tv,
 OD_COEFF tvh) {

  /* +/- Butterflies with asymmetric input. */
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(t0, tv, tvh);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(t1, t1h, tu);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(t2, tt, tth);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(t3, t3h, ts);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(t4, tr, trh);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(t5, t5h, tq);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(t6, tp, tph);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(t7, t7h, to);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(t8, tn, tnh);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(t9, t9h, tm);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(ta, tl, tlh);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tb, tbh, tk);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(tc, tj, tjh);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(td, tdh, ti);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(te, th, thh);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tf, tfh, tg);

  /* Embedded 16-point orthonormal transforms. */
  OD_KERNEL_FUNC(od_fdct_16)(t0, t1, t2, t3, t4, t5, t6, t7,
                             t8, t9, ta, tb, tc, td, te, tf);
  OD_KERNEL_FUNC(od_fdst_16)(tv, tu, tt, ts, tr, tq, tp, to,
                             tn, tm, tl, tk, tj, ti, th, tg);
}

/**
 * 32-point asymmetric Type-II iDCT
 */
static INLINE void OD_KERNEL_FUNC(od_idct_32_asym)(OD_COEFF *t0, OD_COEFF *tg,
 OD_COEFF *t8, OD_COEFF *to, OD_COEFF *t4, OD_COEFF *tk, OD_COEFF *tc,
 OD_COEFF *ts, OD_COEFF *t2, OD_COEFF *ti, OD_COEFF *ta, OD_COEFF *tq,
 OD_COEFF *t6, OD_COEFF *tm, OD_COEFF *te, OD_COEFF *tu, OD_COEFF *t1,
 OD_COEFF *t1h, OD_COEFF *th, OD_COEFF *thh, OD_COEFF *t9, OD_COEFF *t9h,
 OD_COEFF *tp, OD_COEFF *tph, OD_COEFF *t5, OD_COEFF *t5h, OD_COEFF *tl,
 OD_COEFF *tlh, OD_COEFF *td, OD_COEFF *tdh, OD_COEFF *tt, OD_COEFF *tth,
 OD_COEFF *t3, OD_COEFF *t3h, OD_COEFF *tj, OD_COEFF *tjh, OD_COEFF *tb,
 OD_COEFF *tbh, OD_COEFF *tr, OD_COEFF *trh, OD_COEFF *t7, OD_COEFF *t7h,
 OD_COEFF *tn, OD_COEFF *tnh, OD_COEFF *tf, OD_COEFF *tfh, OD_COEFF *tv,
 OD_COEFF *tvh) {

  /* Embedded 16-point orthonormal transforms. */
  OD_KERNEL_FUNC(od_idst_16)(tv, tn, tr, tj, tt, tl, tp, th,
                             tu, tm, tq, ti, ts, tk, to, tg);
  OD_KERNEL_FUNC(od_idct_16)(t0, t8, t4, tc, t2, ta, t6, te,
                             t1, t9, t5, td, t3, tb, t7, tf);

  /* +/- Butterflies with asymmetric output. */
  OD_KERNEL_FUNC(od_butterfly_sub)(tf, tfh, tg);
  OD_KERNEL_FUNC(od_butterfly_neg)(te, th, thh);
  OD_KERNEL_FUNC(od_butterfly_sub)(td, tdh, ti);
  OD_KERNEL_FUNC(od_butterfly_neg)(tc, tj, tjh);
  OD_KERNEL_FUNC(od_butterfly_sub)(tb, tbh, tk);
  OD_KERNEL_FUNC(od_butterfly_neg)(ta, tl, tlh);
  OD_KERNEL_FUNC(od_butterfly_sub)(t9, t9h, tm);
  OD_KERNEL_FUNC(od_butterfly_neg)(t8, tn, tnh);
  OD_KERNEL_FUNC(od_butterfly_sub)(t7, t7h, to);
  OD_KERNEL_FUNC(od_butterfly_neg)(t6, tp, tph);
  OD_KERNEL_FUNC(od_butterfly_sub)(t5, t5h, tq);
  OD_KERNEL_FUNC(od_butterfly_neg)(t4, tr, trh);
  OD_KERNEL_FUNC(od_butterfly_sub)(t3, t3h, ts);
  OD_KERNEL_FUNC(od_butterfly_neg)(t2, tt, tth);
  OD_KERNEL_FUNC(od_butterfly_sub)(t1, t1h, tu);
  OD_KERNEL_FUNC(od_butterfly_neg)(t0, tv, tvh);
}

/**
 * 32-point orthonormal Type-IV fDST
 */
static INLINE void OD_KERNEL_FUNC(od_fdst_32)(OD_COEFF *t0, OD_COEFF *t1,
                                              OD_COEFF *t2, OD_COEFF *t3,
                                              OD_COEFF *t4, OD_COEFF *t5,
                                              OD_COEFF *t6, OD_COEFF *t7,
                                              OD_COEFF *t8, OD_COEFF *t9,
                                              OD_COEFF *ta, OD_COEFF *tb,
                                              OD_COEFF *tc, OD_COEFF *td,
                                              OD_COEFF *te, OD_COEFF *tf,
                                              OD_COEFF *tg, OD_COEFF *th,
                                              OD_COEFF *ti, OD_COEFF *tj,
                                              OD_COEFF *tk, OD_COEFF *tl,
                                              OD_COEFF *tm, OD_COEFF *tn,
                                              OD_COEFF *to, OD_COEFF *tp,
                                              OD_COEFF *tq, OD_COEFF *tr,
                                              OD_COEFF *ts, OD_COEFF *tt,
                                              OD_COEFF *tu, OD_COEFF *tv) {
  OD_COEFF t0h;
  OD_COEFF t1h;
  OD_COEFF t2h;
  OD_COEFF t3h;
  OD_COEFF t4h;
  OD_COEFF t6h;
  OD_COEFF t8h;
  OD_COEFF t9h;
  OD_COEFF tah;
  OD_COEFF tbh;
  OD_COEFF tch;
  OD_COEFF tdh;
  OD_COEFF teh;
  OD_COEFF tfh;
  OD_COEFF tgh;
  OD_COEFF thh;
  OD_COEFF tih;
  OD_COEFF tjh;
  OD_COEFF tkh;
  OD_COEFF tlh;
  OD_COEFF tmh;
  OD_COEFF tnh;
  OD_COEFF tph;
  OD_COEFF trh;
  OD_COEFF tsh;
  OD_COEFF tth;
  OD_COEFF tuh;
  OD_COEFF tvh;

  /* Stage 0 */

  /* Sin[63*Pi/128] + Cos[63*Pi/128] = 1.0242400472191164 */
  /* Sin[63*Pi/128] - Cos[63*Pi/128] = 0.9751575901732919 */
  /* Cos[63*Pi/128]                  = 0.0245412285229123 */
  od_rotate_add(t0, tv, 16781, 14, 15977, 14, 201, 13, TX_NONE);

  /* Sin[61*Pi/128] + Cos[61*Pi/128] = 1.0708550202783576 */
  /* Sin[61*Pi/128] - Cos[61*Pi/128] = 0.9237258930790228 */
  /* Cos[61*Pi/128]                  = 0.0735645635996674 */
  od_rotate_sub(tu, t1, 17545, 14, 30269, 15, 2411, 15, TX_NONE);

  /* Sin[59*Pi/128] + Cos[59*Pi/128] = 1.1148902097979262 */
  /* Sin[59*Pi/128] - Cos[59*Pi/128] = 0.8700688593994937 */
  /* Cos[59*Pi/128]                  = 0.1224106751992162 */
  od_rotate_add(t2, tt, 36533, 15, 14255, 14, 4011, 15, TX_NONE);

  /* Sin[57*Pi/128] + Cos[57*Pi/128] = 1.1562395311492424 */
  /* Sin[57*Pi/128] - Cos[57*Pi/128] = 0.8143157536286401 */
  /* Cos[57*Pi/128]                  = 0.1709618887603012 */
  od_rotate_sub(ts, t3, 37, 5, 26683, 15, 2801, 14, TX_NONE);

  /* Sin[55*Pi/128] + Cos[55*Pi/128] = 1.1948033701953984 */
  /* Sin[55*Pi/128] - Cos[55*Pi/128] = 0.7566008898816587 */
  /* Cos[55*Pi/128]                  = 0.2191012401568698 */
  od_rotate_add(t4, tr, 39151, 15, 3099, 12, 1795, 13, TX_NONE);

  /* Sin[53*Pi/128] + Cos[53*Pi/128] = 1.2304888232703382 */
  /* Sin[53*Pi/128] - Cos[53*Pi/128] = 0.6970633083205415 */
  /* Cos[53*Pi/128]                  = 0.2667127574748984 */
  od_rotate_sub(tq, t5, 40321, 15, 22841, 15, 2185, 13, TX_NONE);

  /* Sin[51*Pi/128] + Cos[51*Pi/128] = 1.2632099209919283 */
  /* Sin[51*Pi/128] - Cos[51*Pi/128] = 0.6358464401941452 */
  /* Cos[51*Pi/128]                  = 0.3136817403988915 */
  od_rotate_add(t6, tp, 41393, 15, 20835, 15, 10279, 15, TX_NONE);

  /* Sin[49*Pi/128] + Cos[49*Pi/128] = 1.2928878353697270 */
  /* Sin[49*Pi/128] - Cos[49*Pi/128] = 0.5730977622997508 */
  /* Cos[49*Pi/128]                  = 0.3598950365349881 */
  od_rotate_sub(to, t7, 42365, 15, 18778, 15, 11793, 15, TX_NONE);

  /* Sin[47*Pi/128] + Cos[47*Pi/128] = 1.3194510697085207 */
  /* Sin[47*Pi/128] - Cos[47*Pi/128] = 0.5089684416985408 */
  /* Cos[47*Pi/128]                  = 0.4052413140049899 */
  od_rotate_add(t8, tn, 10809, 13, 8339, 14, 13279, 15, TX_NONE);

  /* Sin[45*Pi/128] + Cos[45*Pi/128] = 1.3428356308501219 */
  /* Sin[45*Pi/128] - Cos[45*Pi/128] = 0.4436129715409088 */
  /* Cos[45*Pi/128]                  = 0.4496113296546065 */
  od_rotate_sub(tm, t9, 22001, 14, 1817, 12, 14733, 15, TX_NONE);

  /* Sin[43*Pi/128] + Cos[43*Pi/128] = 1.3629851833384956 */
  /* Sin[43*Pi/128] - Cos[43*Pi/128] = 0.3771887988789274 */
  /* Cos[43*Pi/128]                  = 0.4928981922297840 */
  od_rotate_add(ta, tl, 22331, 14, 1545, 12, 16151, 15, TX_NONE);

  /* Sin[41*Pi/128] + Cos[41*Pi/128] = 1.3798511851368043 */
  /* Sin[41*Pi/128] - Cos[41*Pi/128] = 0.3098559453626100 */
  /* Cos[41*Pi/128]                  = 0.5349976198870972 */
  od_rotate_sub(tk, tb, 45215, 15, 10153, 15, 17531, 15, TX_NONE);

  /* Sin[39*Pi/128] + Cos[39*Pi/128] = 1.3933930045694290 */
  /* Sin[39*Pi/128] - Cos[39*Pi/128] = 0.2417766217337384 */
  /* Cos[39*Pi/128]                  = 0.5758081914178453 */
  od_rotate_add(tc, tj, 45659, 15, 7923, 15, 4717, 13, TX_NONE);

  /* Sin[37*Pi/128] + Cos[37*Pi/128] = 1.4035780182072331 */
  /* Sin[37*Pi/128] - Cos[37*Pi/128] = 0.1731148370459795 */
  /* Cos[37*Pi/128]                  = 0.6152315905806268 */
  od_rotate_sub(ti, td, 5749, 12, 5673, 15, 315, 9, TX_NONE);

  /* Sin[35*Pi/128] + Cos[35*Pi/128] = 1.4103816894602614 */
  /* Sin[35*Pi/128] - Cos[35*Pi/128] = 0.1040360035527078 */
  /* Cos[35*Pi/128]                  = 0.6531728429537768 */
  od_rotate_add(te, th, 46215, 15, 3409, 15, 21403, 15, TX_NONE);

  /* Sin[33*Pi/128] + Cos[33*Pi/128] = 1.4137876276885337 */
  /* Sin[33*Pi/128] - Cos[33*Pi/128] = 0.0347065382144002 */
  /* Cos[33*Pi/128]                  = 0.6895405447370668 */
  od_rotate_sub(tg, tf, 46327, 15, 1137, 15, 22595, 15, TX_NONE);

  /* Stage 1 */

  OD_KERNEL_FUNC(od_butterfly_add)(t0, &t0h, tf);
  OD_KERNEL_FUNC(od_butterfly_sub)(tv, &tvh, tg);
  OD_KERNEL_FUNC(od_butterfly_add)(th, &thh, tu);
  OD_KERNEL_FUNC(od_butterfly_sub)(te, &teh, t1);

  OD_KERNEL_FUNC(od_butterfly_add)(t2, &t2h, td);
  OD_KERNEL_FUNC(od_butterfly_sub)(tt, &tth, ti);
  OD_KERNEL_FUNC(od_butterfly_add)(tj, &tjh, ts);
  OD_KERNEL_FUNC(od_butterfly_sub)(tc, &tch, t3);

  OD_KERNEL_FUNC(od_butterfly_add)(t4, &t4h, tb);
  OD_KERNEL_FUNC(od_butterfly_sub)(tr, &trh, tk);
  OD_KERNEL_FUNC(od_butterfly_add)(tl, &tlh, tq);
  OD_KERNEL_FUNC(od_butterfly_sub)(ta, &tah, t5);

  OD_KERNEL_FUNC(od_butterfly_add)(t6, &t6h, t9);
  OD_KERNEL_FUNC(od_butterfly_sub)(tp, &tph, tm);
  OD_KERNEL_FUNC(od_butterfly_add)(tn, &tnh, to);
  OD_KERNEL_FUNC(od_butterfly_sub)(t8, &t8h, t7);

  /* Stage 2 */

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(t0, t0h, t7);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tv, tvh, to);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tp, tph, tu);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t6, t6h, t1);

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(t2, t2h, t5);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tt, tth, tq);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tr, trh, ts);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t4, t4h, t3);

  OD_KERNEL_FUNC(od_butterfly_add_asym)(t8, t8h, tg);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(te, teh, tm);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tn, tnh, tf);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(th, thh, t9);

  OD_KERNEL_FUNC(od_butterfly_add_asym)(ta, tah, ti);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tc, tch, tk);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tl, tlh, td);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tj, tjh, tb);

  /* Stage 3 */

  /* Sin[15*Pi/32] + Cos[15*Pi/32] = 1.0932018670017576 */
  /* Sin[15*Pi/32] - Cos[15*Pi/32] = 0.8971675863426363 */
  /* Cos[15*Pi/32]                 = 0.0980171403295606 */
  od_rotate_sub(tf, tg, 17911, 14, 14699, 14, 803, 13, TX_NONE);

  /* Sin[13*Pi/32] + Cos[13*Pi/32] = 1.2472250129866712 */
  /* Sin[13*Pi/32] - Cos[13*Pi/32] = 0.6666556584777465 */
  /* Cos[13*Pi/32]                 = 0.2902846772544623 */
  od_rotate_add(th, te, 20435, 14, 21845, 15, 1189, 12, TX_NONE);

  /* Sin[11*Pi/32] + Cos[11*Pi/32] = 1.3533180011743526 */
  /* Sin[11*Pi/32] - Cos[11*Pi/32] = 0.4105245275223574 */
  /* Cos[11*Pi/32]                 = 0.4713967368259976 */
  od_rotate_add(ti, td, 22173, 14, 3363, 13, 15447, 15, TX_NONE);

  /* Sin[9*Pi/32] + Cos[9*Pi/32] = 1.4074037375263826 */
  /* Sin[9*Pi/32] - Cos[9*Pi/32] = 0.1386171691990915 */
  /* Cos[9*Pi/32]                = 0.6343932841636455 */
  od_rotate_sub(tc, tj, 23059, 14, 2271, 14, 5197, 13, TX_NONE);

  /* Sin[9*Pi/32] + Cos[9*Pi/32] = 1.4074037375263826 */
  /* Sin[9*Pi/32] - Cos[9*Pi/32] = 0.1386171691990915 */
  /* Cos[9*Pi/32]                = 0.6343932841636455 */
  od_rotate_neg(tb, tk, 23059, 14, 2271, 14, 5197, 13);

  /* Sin[11*Pi/32] + Cos[11*Pi/32] = 1.3533180011743526 */
  /* Sin[11*Pi/32] - Cos[11*Pi/32] = 0.4105245275223574 */
  /* Cos[11*Pi/32]                 = 0.4713967368259976 */
  od_rotate_neg(ta, tl, 22173, 14, 3363, 13, 15447, 15);

  /* Sin[13*Pi/32] + Cos[13*Pi/32] = 1.2472250129866712 */
  /* Sin[13*Pi/32] - Cos[13*Pi/32] = 0.6666556584777465 */
  /* Cos[13*Pi/32]                 = 0.2902846772544623 */
  od_rotate_neg(t9, tm, 20435, 14, 21845, 15, 1189, 12);

  /* Sin[15*Pi/32] + Cos[15*Pi/32] = 1.0932018670017576 */
  /* Sin[15*Pi/32] - Cos[15*Pi/32] = 0.8971675863426363 */
  /* Cos[15*Pi/32]                 = 0.0980171403295606 */
  od_rotate_neg(t8, tn, 17911, 14, 14699, 14, 803, 13);

  /* Stage 4 */

  OD_KERNEL_FUNC(od_butterfly_sub)(t3, &t3h, t0);
  OD_KERNEL_FUNC(od_butterfly_add)(ts, &tsh, tv);
  OD_KERNEL_FUNC(od_butterfly_sub)(tu, &tuh, tt);
  OD_KERNEL_FUNC(od_butterfly_add)(t1, &t1h, t2);

  OD_KERNEL_FUNC(od_butterfly_add)(to, NULL, t4);
  OD_KERNEL_FUNC(od_butterfly_sub)(tq, NULL, t6);
  OD_KERNEL_FUNC(od_butterfly_add)(t7, NULL, tr);
  OD_KERNEL_FUNC(od_butterfly_sub)(t5, NULL, tp);

  OD_KERNEL_FUNC(od_butterfly_sub)(tb, &tbh, t8);
  OD_KERNEL_FUNC(od_butterfly_add)(tk, &tkh, tn);
  OD_KERNEL_FUNC(od_butterfly_sub)(tm, &tmh, tl);
  OD_KERNEL_FUNC(od_butterfly_add)(t9, &t9h, ta);

  OD_KERNEL_FUNC(od_butterfly_sub)(tf, &tfh, tc);
  OD_KERNEL_FUNC(od_butterfly_add)(tg, &tgh, tj);
  OD_KERNEL_FUNC(od_butterfly_sub)(ti, &tih, th);
  OD_KERNEL_FUNC(od_butterfly_add)(td, &tdh, te);

  /* Stage 5 */

  /* Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586 */
  /* Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022 */
  /* Cos[7*Pi/16]                = 0.1950903220161283 */
  od_rotate_add(to, t7, 9633, 13, 12873, 14, 6393, 15, TX_NONE);

  /* Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475 */
  /* Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431 */
  /* Cos[5*Pi/16]                = 0.5555702330196022 */
  od_rotate_add(tp, t6, 22725, 14, 9041, 15, 4551, 13, TX_NONE);

  /* Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475 */
  /* Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431 */
  /* Cos[5*Pi/16]                = 0.5555702330196022 */
  od_rotate_neg(t5, tq, 11363, 13, 9041, 15, 4551, 13);

  /* Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586 */
  /* Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022 */
  /* Cos[7*Pi/16]                = 0.1950903220161283 */
  od_rotate_neg(t4, tr, 9633, 13, 12873, 14, 6393, 15);

  /* Stage 6 */

  OD_KERNEL_FUNC(od_butterfly_add_asym)(t1, t1h, t0);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tu, tuh, tv);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(ts, tsh, t2);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(t3, t3h, tt);

  OD_KERNEL_FUNC(od_butterfly_add_asym)(t5, OD_RSHIFT1(*t5), t4);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tq, OD_RSHIFT1(*tq), tr);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t7, OD_RSHIFT1(*t7), t6);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(to, OD_RSHIFT1(*to), tp);

  OD_KERNEL_FUNC(od_butterfly_add_asym)(t9, t9h, t8);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tm, tmh, tn);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tk, tkh, ta);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tb, tbh, tl);

  OD_KERNEL_FUNC(od_butterfly_add_asym)(ti, tih, tc);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(td, tdh, tj);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tf, tfh, te);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tg, tgh, th);

  /* Stage 7 */

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_neg(t2, tt, 10703, 13, 8867, 14, 3135, 13);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_add(ts, t3, 10703, 13, 8867, 14, 3135, 13, TX_NONE);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_neg(ta, tl, 10703, 13, 8867, 14, 3135, 13);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_add(tk, tb, 10703, 13, 8867, 14, 3135, 13, TX_NONE);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_add(tc, tj, 10703, 13, 8867, 14, 3135, 13, TX_NONE);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_neg(ti, td, 10703, 13, 8867, 14, 3135, 13);

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(tu, t1, 11585, 13, 5793, 13);

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(tq, t5, 11585, 13, 5793, 13);

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_sub(tp, t6, 11585, 13, 5793, 13);

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(tm, t9, 11585, 13, 5793, 13);

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(te, th, 11585, 13, 5793, 13);
}

/**
 * 32-point orthonormal Type-IV fDST
 */
static INLINE void OD_KERNEL_FUNC(od_idst_32)(OD_COEFF *t0, OD_COEFF *tg,
                                              OD_COEFF *t8, OD_COEFF *to,
                                              OD_COEFF *t4, OD_COEFF *tk,
                                              OD_COEFF *tc, OD_COEFF *ts,
                                              OD_COEFF *t2, OD_COEFF *ti,
                                              OD_COEFF *ta, OD_COEFF *tq,
                                              OD_COEFF *t6, OD_COEFF *tm,
                                              OD_COEFF *te, OD_COEFF *tu,
                                              OD_COEFF *t1, OD_COEFF *th,
                                              OD_COEFF *t9, OD_COEFF *tp,
                                              OD_COEFF *t5, OD_COEFF *tl,
                                              OD_COEFF *td, OD_COEFF *tt,
                                              OD_COEFF *t3, OD_COEFF *tj,
                                              OD_COEFF *tb, OD_COEFF *tr,
                                              OD_COEFF *t7, OD_COEFF *tn,
                                              OD_COEFF *tf, OD_COEFF *tv) {
  OD_COEFF t0h;
  OD_COEFF t1h;
  OD_COEFF t2h;
  OD_COEFF t3h;
  OD_COEFF t4h;
  OD_COEFF t6h;
  OD_COEFF t8h;
  OD_COEFF t9h;
  OD_COEFF tah;
  OD_COEFF tbh;
  OD_COEFF tch;
  OD_COEFF tdh;
  OD_COEFF teh;
  OD_COEFF tfh;
  OD_COEFF tgh;
  OD_COEFF thh;
  OD_COEFF tih;
  OD_COEFF tjh;
  OD_COEFF tkh;
  OD_COEFF tlh;
  OD_COEFF tmh;
  OD_COEFF tnh;
  OD_COEFF tph;
  OD_COEFF trh;
  OD_COEFF tsh;
  OD_COEFF tth;
  OD_COEFF tuh;
  OD_COEFF tvh;

  /* Stage 7 */

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(te, th, 11585, 13, 5793, 13);

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(tm, t9, 11585, 13, 5793, 13);

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_sub(tp, t6, 11585, 13, 5793, 13);

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(tq, t5, 11585, 13, 5793, 13);

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(tu, t1, 11585, 13, 5793, 13);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_neg(td, ti, 10703, 13, 8867, 14, 3135, 13);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_add(tc, tj, 10703, 13, 8867, 14, 3135, 13, TX_NONE);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_add(tk, tb, 10703, 13, 8867, 14, 3135, 13, TX_NONE);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_neg(tl, ta, 10703, 13, 8867, 14, 3135, 13);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_add(ts, t3, 10703, 13, 8867, 14, 3135, 13, TX_NONE);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_neg(tt, t2, 10703, 13, 8867, 14, 3135, 13);

  /* Stage 6 */

  OD_KERNEL_FUNC(od_butterfly_sub)(tg, &tgh, th);
  OD_KERNEL_FUNC(od_butterfly_add)(tf, &tfh, te);
  OD_KERNEL_FUNC(od_butterfly_add)(td, &tdh, tj);
  OD_KERNEL_FUNC(od_butterfly_add)(ti, &tih, tc);

  OD_KERNEL_FUNC(od_butterfly_sub)(tb, &tbh, tl);
  OD_KERNEL_FUNC(od_butterfly_sub)(tk, &tkh, ta);
  OD_KERNEL_FUNC(od_butterfly_sub)(tm, &tmh, tn);
  OD_KERNEL_FUNC(od_butterfly_add)(t9, &t9h, t8);

  OD_KERNEL_FUNC(od_butterfly_sub)(to, NULL, tp);
  OD_KERNEL_FUNC(od_butterfly_add)(t7, NULL, t6);
  OD_KERNEL_FUNC(od_butterfly_sub)(tq, NULL, tr);
  OD_KERNEL_FUNC(od_butterfly_add)(t5, NULL, t4);

  OD_KERNEL_FUNC(od_butterfly_sub)(t3, &t3h, tt);
  OD_KERNEL_FUNC(od_butterfly_sub)(ts, &tsh, t2);
  OD_KERNEL_FUNC(od_butterfly_sub)(tu, &tuh, tv);
  OD_KERNEL_FUNC(od_butterfly_add)(t1, &t1h, t0);

  /* Stage 5 */

  /* Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586 */
  /* Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022 */
  /* Cos[7*Pi/16]                = 0.1950903220161283 */
  od_rotate_neg(tr, t4, 9633, 13, 12873, 14, 6393, 15);

  /* Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475 */
  /* Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431 */
  /* Cos[5*Pi/16]                = 0.5555702330196022 */
  od_rotate_neg(tq, t5, 11363, 13, 9041, 15, 4551, 13);

  /* Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475 */
  /* Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431 */
  /* Cos[5*Pi/16]                = 0.5555702330196022 */
  od_rotate_add(tp, t6, 22725, 14, 9041, 15, 4551, 13, TX_NONE);

  /* Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586 */
  /* Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022 */
  /* Cos[7*Pi/16]                = 0.1950903220161283 */
  od_rotate_add(to, t7, 9633, 13, 12873, 14, 6393, 15, TX_NONE);

  /* Stage 4 */

  OD_KERNEL_FUNC(od_butterfly_add_asym)(td, tdh, te);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(ti, tih, th);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tg, tgh, tj);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tf, tfh, tc);

  OD_KERNEL_FUNC(od_butterfly_add_asym)(t9, t9h, ta);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tm, tmh, tl);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tk, tkh, tn);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tb, tbh, t8);

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(t5, OD_RSHIFT1(*t5), tp);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t7, OD_RSHIFT1(*t7), tr);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tq, OD_RSHIFT1(*tq), t6);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(to, OD_RSHIFT1(*to), t4);

  OD_KERNEL_FUNC(od_butterfly_add_asym)(t1, t1h, t2);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tu, tuh, tt);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(ts, tsh, tv);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(t3, t3h, t0);

  /* Stage 3 */

  /* Sin[15*Pi/32] + Cos[15*Pi/32] = 1.0932018670017576 */
  /* Sin[15*Pi/32] - Cos[15*Pi/32] = 0.8971675863426363 */
  /* Cos[15*Pi/32]                 = 0.0980171403295606 */
  od_rotate_neg(tn, t8, 17911, 14, 14699, 14, 803, 13);

  /* Sin[13*Pi/32] + Cos[13*Pi/32] = 1.2472250129866712 */
  /* Sin[13*Pi/32] - Cos[13*Pi/32] = 0.6666556584777465 */
  /* Cos[13*Pi/32]                 = 0.2902846772544623 */
  od_rotate_neg(tm, t9, 20435, 14, 21845, 15, 1189, 12);

  /* Sin[11*Pi/32] + Cos[11*Pi/32] = 1.3533180011743526 */
  /* Sin[11*Pi/32] - Cos[11*Pi/32] = 0.4105245275223574 */
  /* Cos[11*Pi/32]                 = 0.4713967368259976 */
  od_rotate_neg(tl, ta, 22173, 14, 3363, 13, 15447, 15);

  /* Sin[9*Pi/32] + Cos[9*Pi/32] = 1.4074037375263826 */
  /* Sin[9*Pi/32] - Cos[9*Pi/32] = 0.1386171691990915 */
  /* Cos[9*Pi/32]                = 0.6343932841636455 */
  od_rotate_neg(tk, tb, 23059, 14, 2271, 14, 5197, 13);

  /* Sin[9*Pi/32] + Cos[9*Pi/32] = 1.4074037375263826 */
  /* Sin[9*Pi/32] - Cos[9*Pi/32] = 0.1386171691990915 */
  /* Cos[9*Pi/32]                = 0.6343932841636455 */
  od_rotate_sub(tc, tj, 23059, 14, 2271, 14, 5197, 13, TX_NONE);

  /* Sin[11*Pi/32] + Cos[11*Pi/32] = 1.3533180011743526 */
  /* Sin[11*Pi/32] - Cos[11*Pi/32] = 0.4105245275223574 */
  /* Cos[11*Pi/32]                 = 0.4713967368259976 */
  od_rotate_add(ti, td, 22173, 14, 3363, 13, 15447, 15, TX_NONE);

  /* Sin[13*Pi/32] + Cos[13*Pi/32] = 1.2472250129866712 */
  /* Sin[13*Pi/32] - Cos[13*Pi/32] = 0.6666556584777465 */
  /* Cos[13*Pi/32]                 = 0.2902846772544623 */
  od_rotate_add(th, te, 20435, 14, 21845, 15, 1189, 12, TX_NONE);

  /* Sin[15*Pi/32] + Cos[15*Pi/32] = 1.0932018670017576 */
  /* Sin[15*Pi/32] - Cos[15*Pi/32] = 0.8971675863426363 */
  /* Cos[15*Pi/32]                 = 0.0980171403295606 */
  od_rotate_sub(tf, tg, 17911, 14, 14699, 14, 803, 13, TX_NONE);

  /* Stage 2 */

  OD_KERNEL_FUNC(od_butterfly_sub)(tj, &tjh, tb);
  OD_KERNEL_FUNC(od_butterfly_add)(tl, &tlh, td);
  OD_KERNEL_FUNC(od_butterfly_sub)(tc, &tch, tk);
  OD_KERNEL_FUNC(od_butterfly_add)(ta, &tah, ti);

  OD_KERNEL_FUNC(od_butterfly_sub)(th, &thh, t9);
  OD_KERNEL_FUNC(od_butterfly_add)(tn, &tnh, tf);
  OD_KERNEL_FUNC(od_butterfly_sub)(te, &teh, tm);
  OD_KERNEL_FUNC(od_butterfly_add)(t8, &t8h, tg);

  OD_KERNEL_FUNC(od_butterfly_add)(t4, &t4h, t3);
  OD_KERNEL_FUNC(od_butterfly_sub)(tr, &trh, ts);
  OD_KERNEL_FUNC(od_butterfly_add)(tt, &tth, tq);
  OD_KERNEL_FUNC(od_butterfly_sub)(t2, &t2h, t5);

  OD_KERNEL_FUNC(od_butterfly_add)(t6, &t6h, t1);
  OD_KERNEL_FUNC(od_butterfly_sub)(tp, &tph, tu);
  OD_KERNEL_FUNC(od_butterfly_add)(tv, &tvh, to);
  OD_KERNEL_FUNC(od_butterfly_sub)(t0, &t0h, t7);

  /* Stage 1 */

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(t8, t8h, t7);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tn, tnh, to);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tp, tph, tm);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t6, t6h, t9);

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(ta, tah, t5);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tl, tlh, tq);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tr, trh, tk);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t4, t4h, tb);

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tc, tch, t3);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tj, tjh, ts);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tt, tth, ti);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t2, t2h, td);

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(te, teh, t1);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(th, thh, tu);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tv, tvh, tg);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t0, t0h, tf);

  /* Stage 0 */

  /* Sin[33*Pi/128] + Cos[33*Pi/128] = 1.4137876276885337 */
  /* Sin[33*Pi/128] - Cos[33*Pi/128] = 0.0347065382144002 */
  /* Cos[33*Pi/128]                  = 0.6895405447370668 */
  od_rotate_sub(tg, tf, 46327, 15, 1137, 15, 22595, 15, TX_NONE);

  /* Sin[35*Pi/128] + Cos[35*Pi/128] = 1.4103816894602614 */
  /* Sin[35*Pi/128] - Cos[35*Pi/128] = 0.1040360035527078 */
  /* Cos[35*Pi/128]                  = 0.6531728429537768 */
  od_rotate_add(te, th, 46215, 15, 3409, 15, 21403, 15, TX_NONE);

  /* Sin[37*Pi/128] + Cos[37*Pi/128] = 1.4035780182072331 */
  /* Sin[37*Pi/128] - Cos[37*Pi/128] = 0.1731148370459795 */
  /* Cos[37*Pi/128]                  = 0.6152315905806268 */
  od_rotate_sub(ti, td, 5749, 12, 5673, 15, 315, 9, TX_NONE);

  /* Sin[39*Pi/128] + Cos[39*Pi/128] = 1.3933930045694290 */
  /* Sin[39*Pi/128] - Cos[39*Pi/128] = 0.2417766217337384 */
  /* Cos[39*Pi/128]                  = 0.5758081914178453 */
  od_rotate_add(tc, tj, 45659, 15, 7923, 15, 4717, 13, TX_NONE);

  /* Sin[41*Pi/128] + Cos[41*Pi/128] = 1.3798511851368043 */
  /* Sin[41*Pi/128] - Cos[41*Pi/128] = 0.3098559453626100 */
  /* Cos[41*Pi/128]                  = 0.5349976198870972 */
  od_rotate_sub(tk, tb, 45215, 15, 10153, 15, 17531, 15, TX_NONE);

  /* Sin[43*Pi/128] + Cos[43*Pi/128] = 1.3629851833384956 */
  /* Sin[43*Pi/128] - Cos[43*Pi/128] = 0.3771887988789274 */
  /* Cos[43*Pi/128]                  = 0.4928981922297840 */
  od_rotate_add(ta, tl, 22331, 14, 1545, 12, 16151, 15, TX_NONE);

  /* Sin[45*Pi/128] + Cos[45*Pi/128] = 1.3428356308501219 */
  /* Sin[45*Pi/128] - Cos[45*Pi/128] = 0.4436129715409088 */
  /* Cos[45*Pi/128]                  = 0.4496113296546065 */
  od_rotate_sub(tm, t9, 22001, 14, 1817, 12, 14733, 15, TX_NONE);

  /* Sin[47*Pi/128] + Cos[47*Pi/128] = 1.3194510697085207 */
  /* Sin[47*Pi/128] - Cos[47*Pi/128] = 0.5089684416985408 */
  /* Cos[47*Pi/128]                  = 0.4052413140049899 */
  od_rotate_add(t8, tn, 10809, 13, 8339, 14, 13279, 15, TX_NONE);

  /* Sin[49*Pi/128] + Cos[49*Pi/128] = 1.2928878353697270 */
  /* Sin[49*Pi/128] - Cos[49*Pi/128] = 0.5730977622997508 */
  /* Cos[49*Pi/128]                  = 0.3598950365349881 */
  od_rotate_sub(to, t7, 42365, 15, 18779, 15, 11793, 15, TX_NONE);

  /* Sin[51*Pi/128] + Cos[51*Pi/128] = 1.2632099209919283 */
  /* Sin[51*Pi/128] - Cos[51*Pi/128] = 0.6358464401941452 */
  /* Cos[51*Pi/128]                  = 0.3136817403988915 */
  od_rotate_add(t6, tp, 41393, 15, 20835, 15, 10279, 15, TX_NONE);

  /* Sin[53*Pi/128] + Cos[53*Pi/128] = 1.2304888232703382 */
  /* Sin[53*Pi/128] - Cos[53*Pi/128] = 0.6970633083205415 */
  /* Cos[53*Pi/128]                  = 0.2667127574748984 */
  od_rotate_sub(tq, t5, 40321, 15, 22841, 15, 2185, 13, TX_NONE);

  /* Sin[55*Pi/128] + Cos[55*Pi/128] = 1.1948033701953984 */
  /* Sin[55*Pi/128] - Cos[55*Pi/128] = 0.7566008898816587 */
  /* Cos[55*Pi/128]                  = 0.2191012401568698 */
  od_rotate_add(t4, tr, 39151, 15, 3099, 12, 1795, 13, TX_NONE);

  /* Sin[57*Pi/128] + Cos[57*Pi/128] = 1.1562395311492424 */
  /* Sin[57*Pi/128] - Cos[57*Pi/128] = 0.8143157536286401 */
  /* Cos[57*Pi/128]                  = 0.1709618887603012 */
  od_rotate_sub(ts, t3, 37, 5, 26683, 15, 2801, 14, TX_NONE);

  /* Sin[59*Pi/128] + Cos[59*Pi/128] = 1.1148902097979262 */
  /* Sin[59*Pi/128] - Cos[59*Pi/128] = 0.8700688593994937 */
  /* Cos[59*Pi/128]                  = 0.1224106751992162 */
  od_rotate_add(t2, tt, 36533, 15, 14255, 14, 4011, 15, TX_NONE);

  /* Sin[61*Pi/128] + Cos[61*Pi/128] = 1.0708550202783576 */
  /* Sin[61*Pi/128] - Cos[61*Pi/128] = 0.9237258930790228 */
  /* Cos[61*Pi/128]                  = 0.0735645635996674 */
  od_rotate_sub(tu, t1, 17545, 14, 30269, 15, 2411, 15, TX_NONE);

  /* Sin[63*Pi/128] + Cos[63*Pi/128] = 1.0242400472191164 */
  /* Sin[63*Pi/128] - Cos[63*Pi/128] = 0.9751575901732919 */
  /* Cos[63*Pi/128]                  = 0.0245412285229123 */
  od_rotate_add(t0, tv, 16781, 14, 15977, 14, 201, 13, TX_NONE);
}

static INLINE void OD_KERNEL_FUNC(od_flip_idst_32)(OD_COEFF *t0,
 OD_COEFF *tg, OD_COEFF *t8, OD_COEFF *to, OD_COEFF *t4, OD_COEFF *tk,
 OD_COEFF *tc, OD_COEFF *ts, OD_COEFF *t2, OD_COEFF *ti, OD_COEFF *ta,
 OD_COEFF *tq, OD_COEFF *t6, OD_COEFF *tm, OD_COEFF *te, OD_COEFF *tu,
 OD_COEFF *t1, OD_COEFF *th, OD_COEFF *t9, OD_COEFF *tp, OD_COEFF *t5,
 OD_COEFF *tl, OD_COEFF *td, OD_COEFF *tt, OD_COEFF *t3, OD_COEFF *tj,
 OD_COEFF *tb, OD_COEFF *tr, OD_COEFF *t7, OD_COEFF *tn, OD_COEFF *tf,
 OD_COEFF *tv) {
  OD_KERNEL_FUNC(od_idst_32)(
   t0, tg, t8, to, t4, tk, tc, ts, t2, ti, ta, tq, t6, tm, te, tu,
   t1, th, t9, tp, t5, tl, td, tt, t3, tj, tb, tr, t7, tn, tf, tv);
  OD_SWAP(t0, tv);
  OD_SWAP(tg, tf);
  OD_SWAP(t8, tn);
  OD_SWAP(to, t7);
  OD_SWAP(t4, tr);
  OD_SWAP(tk, tb);
  OD_SWAP(tc, tj);
  OD_SWAP(ts, t3);
  OD_SWAP(t2, tt);
  OD_SWAP(ti, td);
  OD_SWAP(ta, tl);
  OD_SWAP(tq, t5);
  OD_SWAP(t6, tp);
  OD_SWAP(tm, t9);
  OD_SWAP(te, th);
  OD_SWAP(tu, t1);
}

/**
 * 32-point orthonormal Type-IV fDST
 */
static INLINE void OD_KERNEL_FUNC(od_fdst_32_asym)(OD_COEFF *t0, OD_COEFF t0h,
 OD_COEFF *t1, OD_COEFF *t2, OD_COEFF t2h, OD_COEFF *t3, OD_COEFF *t4,
 OD_COEFF t4h, OD_COEFF *t5, OD_COEFF *t6, OD_COEFF t6h, OD_COEFF *t7,
 OD_COEFF *t8, OD_COEFF t8h, OD_COEFF *t9, OD_COEFF *ta, OD_COEFF tah,
 OD_COEFF *tb, OD_COEFF *tc, OD_COEFF tch, OD_COEFF *td, OD_COEFF *te,
 OD_COEFF teh, OD_COEFF *tf, OD_COEFF *tg, OD_COEFF tgh, OD_COEFF *th,
 OD_COEFF *ti, OD_COEFF tih, OD_COEFF *tj, OD_COEFF *tk, OD_COEFF tkh,
 OD_COEFF *tl, OD_COEFF *tm, OD_COEFF tmh, OD_COEFF *tn, OD_COEFF *to,
 OD_COEFF toh, OD_COEFF *tp, OD_COEFF *tq, OD_COEFF tqh, OD_COEFF *tr,
 OD_COEFF *ts, OD_COEFF tsh, OD_COEFF *tt, OD_COEFF *tu, OD_COEFF tuh,
 OD_COEFF *tv) {
  OD_COEFF t1h;
  OD_COEFF t3h;
  OD_COEFF t9h;
  OD_COEFF tbh;
  OD_COEFF tdh;
  OD_COEFF tfh;
  OD_COEFF thh;
  OD_COEFF tjh;
  OD_COEFF tlh;
  OD_COEFF tnh;
  OD_COEFF tph;
  OD_COEFF trh;
  OD_COEFF tth;
  OD_COEFF tvh;

  /* Stage 0 */

  /* Sin[63*Pi/128] + Cos[63*Pi/128] = 1.0242400472191164 */
  /* Sin[63*Pi/128] - Cos[63*Pi/128] = 0.9751575901732919 */
  /* Cos[63*Pi/128]                  = 0.0245412285229123 */
  od_rotate_add_half(t0, tv, t0h, 5933, 13, 22595, 14, 1137, 15, TX_NONE);

  /* Sin[61*Pi/128] + Cos[61*Pi/128] = 1.0708550202783576 */
  /* Sin[61*Pi/128] - Cos[61*Pi/128] = 0.9237258930790228 */
  /* Cos[61*Pi/128]                  = 0.0735645635996674 */
  od_rotate_sub_half(tu, t1, tuh, 6203, 13, 21403, 14, 3409, 15, TX_NONE);

  /* Sin[59*Pi/128] + Cos[59*Pi/128] = 1.1148902097979262 */
  /* Sin[59*Pi/128] - Cos[59*Pi/128] = 0.8700688593994937 */
  /* Cos[59*Pi/128]                  = 0.1224106751992162 */
  od_rotate_add_half(t2, tt, t2h, 25833, 15, 315, 8, 5673, 15, TX_NONE);

  /* Sin[57*Pi/128] + Cos[57*Pi/128] = 1.1562395311492424 */
  /* Sin[57*Pi/128] - Cos[57*Pi/128] = 0.8143157536286401 */
  /* Cos[57*Pi/128]                  = 0.1709618887603012 */
  od_rotate_sub_half(ts, t3, tsh, 26791, 15, 4717, 12, 7923, 15, TX_NONE);

  /* Sin[55*Pi/128] + Cos[55*Pi/128] = 1.1948033701953984 */
  /* Sin[55*Pi/128] - Cos[55*Pi/128] = 0.7566008898816587 */
  /* Cos[55*Pi/128]                  = 0.2191012401568698 */
  od_rotate_add_half(t4, tr, t4h, 6921, 13, 17531, 14, 10153, 15, TX_NONE);

  /* Sin[53*Pi/128] + Cos[53*Pi/128] = 1.2304888232703382 */
  /* Sin[53*Pi/128] - Cos[53*Pi/128] = 0.6970633083205415 */
  /* Cos[53*Pi/128]                  = 0.2667127574748984 */
  od_rotate_sub_half(tq, t5, tqh, 28511, 15, 32303, 15, 1545, 12, TX_NONE);

  /* Sin[51*Pi/128] + Cos[51*Pi/128] = 1.2632099209919283 */
  /* Sin[51*Pi/128] - Cos[51*Pi/128] = 0.6358464401941452 */
  /* Cos[51*Pi/128]                  = 0.3136817403988915 */
  od_rotate_add_half(t6, tp, t6h, 29269, 15, 14733, 14, 1817, 12, TX_NONE);

  /* Sin[49*Pi/128] + Cos[49*Pi/128] = 1.2928878353697270 */
  /* Sin[49*Pi/128] - Cos[49*Pi/128] = 0.5730977622997508 */
  /* Cos[49*Pi/128]                  = 0.3598950365349881 */
  od_rotate_sub_half(to, t7, toh, 29957, 15, 13279, 14, 8339, 14, TX_NONE);

  /* Sin[47*Pi/128] + Cos[47*Pi/128] = 1.3194510697085207 */
  /* Sin[47*Pi/128] - Cos[47*Pi/128] = 0.5089684416985408 */
  /* Cos[47*Pi/128]                  = 0.4052413140049899 */
  od_rotate_add_half(t8, tn, t8h, 7643, 13, 11793, 14, 18779, 15, TX_NONE);

  /* Sin[45*Pi/128] + Cos[45*Pi/128] = 1.3428356308501219 */
  /* Sin[45*Pi/128] - Cos[45*Pi/128] = 0.4436129715409088 */
  /* Cos[45*Pi/128]                  = 0.4496113296546065 */
  od_rotate_sub_half(tm, t9, tmh, 15557, 14, 20557, 15, 20835, 15, TX_NONE);

  /* Sin[43*Pi/128] + Cos[43*Pi/128] = 1.3629851833384956 */
  /* Sin[43*Pi/128] - Cos[43*Pi/128] = 0.3771887988789274 */
  /* Cos[43*Pi/128]                  = 0.4928981922297840 */
  od_rotate_add_half(ta, tl, tah, 31581, 15, 17479, 15, 22841, 15, TX_NONE);

  /* Sin[41*Pi/128] + Cos[41*Pi/128] = 1.3798511851368043 */
  /* Sin[41*Pi/128] - Cos[41*Pi/128] = 0.3098559453626100 */
  /* Cos[41*Pi/128]                  = 0.5349976198870972 */
  od_rotate_sub_half(tk, tb, tkh, 7993, 13, 14359, 15, 3099, 12, TX_NONE);

  /* Sin[39*Pi/128] + Cos[39*Pi/128] = 1.3933930045694290 */
  /* Sin[39*Pi/128] - Cos[39*Pi/128] = 0.2417766217337384 */
  /* Cos[39*Pi/128]                  = 0.5758081914178453 */
  od_rotate_add_half(tc, tj, tch, 16143, 14, 2801, 13, 26683, 15, TX_NONE);

  /* Sin[37*Pi/128] + Cos[37*Pi/128] = 1.4035780182072331 */
  /* Sin[37*Pi/128] - Cos[37*Pi/128] = 0.1731148370459795 */
  /* Cos[37*Pi/128]                  = 0.6152315905806268 */
  od_rotate_sub_half(ti, td, tih, 16261, 14, 4011, 14, 14255, 14, TX_NONE);

  /* Sin[35*Pi/128] + Cos[35*Pi/128] = 1.4103816894602614 */
  /* Sin[35*Pi/128] - Cos[35*Pi/128] = 0.1040360035527078 */
  /* Cos[35*Pi/128]                  = 0.6531728429537768 */
  od_rotate_add_half(te, th, teh, 32679, 15, 4821, 15, 30269, 15, TX_NONE);

  /* Sin[33*Pi/128] + Cos[33*Pi/128] = 1.4137876276885337 */
  /* Sin[33*Pi/128] - Cos[33*Pi/128] = 0.0347065382144002 */
  /* Cos[33*Pi/128]                  = 0.6895405447370668 */
  od_rotate_sub_half(tg, tf, tgh, 16379, 14, 201, 12, 15977, 14, TX_NONE);

  /* Stage 1 */

  OD_KERNEL_FUNC(od_butterfly_add)(t0, &t0h, tf);
  OD_KERNEL_FUNC(od_butterfly_sub)(tv, &tvh, tg);
  OD_KERNEL_FUNC(od_butterfly_add)(th, &thh, tu);
  OD_KERNEL_FUNC(od_butterfly_sub)(te, &teh, t1);

  OD_KERNEL_FUNC(od_butterfly_add)(t2, &t2h, td);
  OD_KERNEL_FUNC(od_butterfly_sub)(tt, &tth, ti);
  OD_KERNEL_FUNC(od_butterfly_add)(tj, &tjh, ts);
  OD_KERNEL_FUNC(od_butterfly_sub)(tc, &tch, t3);

  OD_KERNEL_FUNC(od_butterfly_add)(t4, &t4h, tb);
  OD_KERNEL_FUNC(od_butterfly_sub)(tr, &trh, tk);
  OD_KERNEL_FUNC(od_butterfly_add)(tl, &tlh, tq);
  OD_KERNEL_FUNC(od_butterfly_sub)(ta, &tah, t5);

  OD_KERNEL_FUNC(od_butterfly_add)(t6, &t6h, t9);
  OD_KERNEL_FUNC(od_butterfly_sub)(tp, &tph, tm);
  OD_KERNEL_FUNC(od_butterfly_add)(tn, &tnh, to);
  OD_KERNEL_FUNC(od_butterfly_sub)(t8, &t8h, t7);

  /* Stage 2 */

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(t0, t0h, t7);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tv, tvh, to);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tp, tph, tu);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t6, t6h, t1);

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(t2, t2h, t5);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tt, tth, tq);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tr, trh, ts);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t4, t4h, t3);

  OD_KERNEL_FUNC(od_butterfly_add_asym)(t8, t8h, tg);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(te, teh, tm);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tn, tnh, tf);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(th, thh, t9);

  OD_KERNEL_FUNC(od_butterfly_add_asym)(ta, tah, ti);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tc, tch, tk);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tl, tlh, td);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tj, tjh, tb);

  /* Stage 3 */

  /* Sin[15*Pi/32] + Cos[15*Pi/32] = 1.0932018670017576 */
  /* Sin[15*Pi/32] - Cos[15*Pi/32] = 0.8971675863426363 */
  /* Cos[15*Pi/32]                 = 0.0980171403295606 */
  od_rotate_sub(tf, tg, 17911, 14, 14699, 14, 803, 13, TX_NONE);

  /* Sin[13*Pi/32] + Cos[13*Pi/32] = 1.2472250129866712 */
  /* Sin[13*Pi/32] - Cos[13*Pi/32] = 0.6666556584777465 */
  /* Cos[13*Pi/32]                 = 0.2902846772544623 */
  od_rotate_add(th, te, 10217, 13, 5461, 13, 1189, 12, TX_NONE);

  /* Sin[11*Pi/32] + Cos[11*Pi/32] = 1.3533180011743526 */
  /* Sin[11*Pi/32] - Cos[11*Pi/32] = 0.4105245275223574 */
  /* Cos[11*Pi/32]                 = 0.4713967368259976 */
  od_rotate_add(ti, td, 5543, 12, 3363, 13, 7723, 14, TX_NONE);

  /* Sin[9*Pi/32] + Cos[9*Pi/32] = 1.4074037375263826 */
  /* Sin[9*Pi/32] - Cos[9*Pi/32] = 0.1386171691990915 */
  /* Cos[9*Pi/32]                = 0.6343932841636455 */
  od_rotate_sub(tc, tj, 11529, 13, 2271, 14, 5197, 13, TX_NONE);

  /* Sin[9*Pi/32] + Cos[9*Pi/32] = 1.4074037375263826 */
  /* Sin[9*Pi/32] - Cos[9*Pi/32] = 0.1386171691990915 */
  /* Cos[9*Pi/32]                = 0.6343932841636455 */
  od_rotate_neg(tb, tk, 11529, 13, 2271, 14, 5197, 13);

  /* Sin[11*Pi/32] + Cos[11*Pi/32] = 1.3533180011743526 */
  /* Sin[11*Pi/32] - Cos[11*Pi/32] = 0.4105245275223574 */
  /* Cos[11*Pi/32]                 = 0.4713967368259976 */
  od_rotate_neg(ta, tl, 5543, 12, 3363, 13, 7723, 14);

  /* Sin[13*Pi/32] + Cos[13*Pi/32] = 1.2472250129866712 */
  /* Sin[13*Pi/32] - Cos[13*Pi/32] = 0.6666556584777465 */
  /* Cos[13*Pi/32]                 = 0.2902846772544623 */
  od_rotate_neg(t9, tm, 10217, 13, 5461, 13, 1189, 12);

  /* Sin[15*Pi/32] + Cos[15*Pi/32] = 1.0932018670017576 */
  /* Sin[15*Pi/32] - Cos[15*Pi/32] = 0.8971675863426363 */
  /* Cos[15*Pi/32]                 = 0.0980171403295606 */
  od_rotate_neg(t8, tn, 17911, 14, 14699, 14, 803, 13);

  /* Stage 4 */

  OD_KERNEL_FUNC(od_butterfly_sub)(t3, &t3h, t0);
  OD_KERNEL_FUNC(od_butterfly_add)(ts, &tsh, tv);
  OD_KERNEL_FUNC(od_butterfly_sub)(tu, &tuh, tt);
  OD_KERNEL_FUNC(od_butterfly_add)(t1, &t1h, t2);

  OD_KERNEL_FUNC(od_butterfly_add)(to, NULL, t4);
  OD_KERNEL_FUNC(od_butterfly_sub)(tq, NULL, t6);
  OD_KERNEL_FUNC(od_butterfly_add)(t7, NULL, tr);
  OD_KERNEL_FUNC(od_butterfly_sub)(t5, NULL, tp);

  OD_KERNEL_FUNC(od_butterfly_sub)(tb, &tbh, t8);
  OD_KERNEL_FUNC(od_butterfly_add)(tk, &tkh, tn);
  OD_KERNEL_FUNC(od_butterfly_sub)(tm, &tmh, tl);
  OD_KERNEL_FUNC(od_butterfly_add)(t9, &t9h, ta);

  OD_KERNEL_FUNC(od_butterfly_sub)(tf, &tfh, tc);
  OD_KERNEL_FUNC(od_butterfly_add)(tg, &tgh, tj);
  OD_KERNEL_FUNC(od_butterfly_sub)(ti, &tih, th);
  OD_KERNEL_FUNC(od_butterfly_add)(td, &tdh, te);

  /* Stage 5 */

  /* Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586 */
  /* Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022 */
  /* Cos[7*Pi/16]                = 0.1950903220161283 */
  od_rotate_add(to, t7, 301, 8, 1609, 11, 6393, 15, TX_NONE);

  /* Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475 */
  /* Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431 */
  /* Cos[5*Pi/16]                = 0.5555702330196022 */
  od_rotate_add(tp, t6, 11363, 13, 9041, 15, 4551, 13, TX_NONE);

  /* Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475 */
  /* Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431 */
  /* Cos[5*Pi/16]                = 0.5555702330196022 */
  od_rotate_neg(t5, tq, 5681, 12, 9041, 15, 4551, 13);

  /* Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586 */
  /* Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022 */
  /* Cos[7*Pi/16]                = 0.1950903220161283 */
  od_rotate_neg(t4, tr, 9633, 13, 12873, 14, 6393, 15);

  /* Stage 6 */

  OD_KERNEL_FUNC(od_butterfly_add_asym)(t1, t1h, t0);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tu, tuh, tv);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(ts, tsh, t2);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(t3, t3h, tt);

  OD_KERNEL_FUNC(od_butterfly_add_asym)(t5, OD_RSHIFT1(*t5), t4);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tq, OD_RSHIFT1(*tq), tr);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t7, OD_RSHIFT1(*t7), t6);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(to, OD_RSHIFT1(*to), tp);

  OD_KERNEL_FUNC(od_butterfly_add_asym)(t9, t9h, t8);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tm, tmh, tn);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tk, tkh, ta);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tb, tbh, tl);

  OD_KERNEL_FUNC(od_butterfly_add_asym)(ti, tih, tc);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(td, tdh, tj);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tf, tfh, te);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tg, tgh, th);

  /* Stage 7 */

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_neg(t2, tt, 669, 9, 8867, 14, 3135, 13);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_add(ts, t3, 669, 9, 8867, 14, 3135, 13, TX_NONE);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_neg(ta, tl, 669, 9, 8867, 14, 3135, 13);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_add(tk, tb, 669, 9, 8867, 14, 3135, 13, TX_NONE);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_add(tc, tj, 669, 9, 8867, 14, 3135, 13, TX_NONE);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_neg(ti, td, 669, 9, 8867, 14, 3135, 13);

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(tu, t1, 5793, 12, 5793, 13);

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(tq, t5, 5793, 12, 5793, 13);

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_sub(tp, t6, 5793, 12, 5793, 13);

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(tm, t9, 5793, 12, 5793, 13);

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(te, th, 5793, 12, 5793, 13);
}

/**
 * 32-point asymmetric Type-IV iDST
 */
static INLINE void OD_KERNEL_FUNC(od_idst_32_asym)(OD_COEFF *t0, OD_COEFF *tg,
 OD_COEFF *t8, OD_COEFF *to, OD_COEFF *t4, OD_COEFF *tk, OD_COEFF *tc,
 OD_COEFF *ts, OD_COEFF *t2, OD_COEFF *ti, OD_COEFF *ta, OD_COEFF *tq,
 OD_COEFF *t6, OD_COEFF *tm, OD_COEFF *te, OD_COEFF *tu, OD_COEFF *t1,
 OD_COEFF *th, OD_COEFF *t9, OD_COEFF *tp, OD_COEFF *t5, OD_COEFF *tl,
 OD_COEFF *td, OD_COEFF *tt, OD_COEFF *t3, OD_COEFF *tj, OD_COEFF *tb,
 OD_COEFF *tr, OD_COEFF *t7, OD_COEFF *tn, OD_COEFF *tf, OD_COEFF *tv) {
  OD_COEFF t0h;
  OD_COEFF t1h;
  OD_COEFF t2h;
  OD_COEFF t3h;
  OD_COEFF t4h;
  OD_COEFF t6h;
  OD_COEFF t8h;
  OD_COEFF t9h;
  OD_COEFF tah;
  OD_COEFF tbh;
  OD_COEFF tch;
  OD_COEFF tdh;
  OD_COEFF teh;
  OD_COEFF tfh;
  OD_COEFF tgh;
  OD_COEFF thh;
  OD_COEFF tih;
  OD_COEFF tjh;
  OD_COEFF tkh;
  OD_COEFF tlh;
  OD_COEFF tmh;
  OD_COEFF tnh;
  OD_COEFF tph;
  OD_COEFF trh;
  OD_COEFF tsh;
  OD_COEFF tth;
  OD_COEFF tuh;
  OD_COEFF tvh;

  /* Stage 7 */

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(te, th, 5793, 12, 5793, 13);

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(tm, t9, 5793, 12, 5793, 13);

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_sub(tp, t6, 5793, 12, 5793, 13);

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(tq, t5, 5793, 12, 5793, 13);

  /* Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951 */
  /* Cos[Pi/4]             = 0.7071067811865475 */
  od_rotate_pi4_add(tu, t1, 5793, 12, 5793, 13);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_neg(td, ti, 669, 9, 8867, 14, 3135, 13);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_add(tc, tj, 669, 9, 8867, 14, 3135, 13, TX_NONE);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_add(tk, tb, 669, 9, 8867, 14, 3135, 13, TX_NONE);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_neg(tl, ta, 669, 9, 8867, 14, 3135, 13);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_add(ts, t3, 669, 9, 8867, 14, 3135, 13, TX_NONE);

  /* Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766 */
  /* Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969 */
  /* Cos[3*Pi/8]               = 0.3826834323650898 */
  od_rotate_neg(tt, t2, 669, 9, 8867, 14, 3135, 13);

  /* Stage 6 */

  OD_KERNEL_FUNC(od_butterfly_sub)(tg, &tgh, th);
  OD_KERNEL_FUNC(od_butterfly_add)(tf, &tfh, te);
  OD_KERNEL_FUNC(od_butterfly_add)(td, &tdh, tj);
  OD_KERNEL_FUNC(od_butterfly_add)(ti, &tih, tc);

  OD_KERNEL_FUNC(od_butterfly_sub)(tb, &tbh, tl);
  OD_KERNEL_FUNC(od_butterfly_sub)(tk, &tkh, ta);
  OD_KERNEL_FUNC(od_butterfly_sub)(tm, &tmh, tn);
  OD_KERNEL_FUNC(od_butterfly_add)(t9, &t9h, t8);

  OD_KERNEL_FUNC(od_butterfly_sub)(to, NULL, tp);
  OD_KERNEL_FUNC(od_butterfly_add)(t7, NULL, t6);
  OD_KERNEL_FUNC(od_butterfly_sub)(tq, NULL, tr);
  OD_KERNEL_FUNC(od_butterfly_add)(t5, NULL, t4);

  OD_KERNEL_FUNC(od_butterfly_sub)(t3, &t3h, tt);
  OD_KERNEL_FUNC(od_butterfly_sub)(ts, &tsh, t2);
  OD_KERNEL_FUNC(od_butterfly_sub)(tu, &tuh, tv);
  OD_KERNEL_FUNC(od_butterfly_add)(t1, &t1h, t0);

  /* Stage 5 */

  /* Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586 */
  /* Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022 */
  /* Cos[7*Pi/16]                = 0.1950903220161283 */
  od_rotate_neg(tr, t4, 9633, 13, 12873, 14, 6393, 15);

  /* Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475 */
  /* Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431 */
  /* Cos[5*Pi/16]                = 0.5555702330196022 */
  od_rotate_neg(tq, t5, 5681, 12, 9041, 15, 4551, 13);

  /* Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475 */
  /* Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431 */
  /* Cos[5*Pi/16]                = 0.5555702330196022 */
  od_rotate_add(tp, t6, 11363, 13, 9041, 15, 4551, 13, TX_NONE);

  /* Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586 */
  /* Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022 */
  /* Cos[7*Pi/16]                = 0.1950903220161283 */
  od_rotate_add(to, t7, 301, 8, 1609, 11, 6393, 15, TX_NONE);

  /* Stage 4 */

  OD_KERNEL_FUNC(od_butterfly_add_asym)(td, tdh, te);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(ti, tih, th);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tg, tgh, tj);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tf, tfh, tc);

  OD_KERNEL_FUNC(od_butterfly_add_asym)(t9, t9h, ta);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tm, tmh, tl);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tk, tkh, tn);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tb, tbh, t8);

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(t5, OD_RSHIFT1(*t5), tp);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t7, OD_RSHIFT1(*t7), tr);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tq, OD_RSHIFT1(*tq), t6);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(to, OD_RSHIFT1(*to), t4);

  OD_KERNEL_FUNC(od_butterfly_add_asym)(t1, t1h, t2);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tu, tuh, tt);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(ts, tsh, tv);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(t3, t3h, t0);

  /* Stage 3 */

  /* Sin[15*Pi/32] + Cos[15*Pi/32] = 1.0932018670017576 */
  /* Sin[15*Pi/32] - Cos[15*Pi/32] = 0.8971675863426363 */
  /* Cos[15*Pi/32]                 = 0.0980171403295606 */
  od_rotate_neg(tn, t8, 17911, 14, 14699, 14, 803, 13);

  /* Sin[13*Pi/32] + Cos[13*Pi/32] = 1.2472250129866712 */
  /* Sin[13*Pi/32] - Cos[13*Pi/32] = 0.6666556584777465 */
  /* Cos[13*Pi/32]                 = 0.2902846772544623 */
  od_rotate_neg(tm, t9, 10217, 13, 5461, 13, 1189, 12);

  /* Sin[11*Pi/32] + Cos[11*Pi/32] = 1.3533180011743526 */
  /* Sin[11*Pi/32] - Cos[11*Pi/32] = 0.4105245275223574 */
  /* Cos[11*Pi/32]                 = 0.4713967368259976 */
  od_rotate_neg(tl, ta, 5543, 12, 3363, 13, 7723, 14);

  /* Sin[9*Pi/32] + Cos[9*Pi/32] = 1.4074037375263826 */
  /* Sin[9*Pi/32] - Cos[9*Pi/32] = 0.1386171691990915 */
  /* Cos[9*Pi/32]                = 0.6343932841636455 */
  od_rotate_neg(tk, tb, 11529, 13, 2271, 14, 5197, 13);

  /* Sin[9*Pi/32] + Cos[9*Pi/32] = 1.4074037375263826 */
  /* Sin[9*Pi/32] - Cos[9*Pi/32] = 0.1386171691990915 */
  /* Cos[9*Pi/32]                = 0.6343932841636455 */
  od_rotate_sub(tc, tj, 11529, 13, 2271, 14, 5197, 13, TX_NONE);

  /* Sin[11*Pi/32] + Cos[11*Pi/32] = 1.3533180011743526 */
  /* Sin[11*Pi/32] - Cos[11*Pi/32] = 0.4105245275223574 */
  /* Cos[11*Pi/32]                 = 0.4713967368259976 */
  od_rotate_add(ti, td, 5543, 12, 3363, 13, 7723, 14, TX_NONE);

  /* Sin[13*Pi/32] + Cos[13*Pi/32] = 1.2472250129866712 */
  /* Sin[13*Pi/32] - Cos[13*Pi/32] = 0.6666556584777465 */
  /* Cos[13*Pi/32]                 = 0.2902846772544623 */
  od_rotate_add(th, te, 10217, 13, 5461, 13, 1189, 12, TX_NONE);

  /* Sin[15*Pi/32] + Cos[15*Pi/32] = 1.0932018670017576 */
  /* Sin[15*Pi/32] - Cos[15*Pi/32] = 0.8971675863426363 */
  /* Cos[15*Pi/32]                 = 0.0980171403295606 */
  od_rotate_sub(tf, tg, 17911, 14, 14699, 14, 803, 13, TX_NONE);

  /* Stage 2 */

  OD_KERNEL_FUNC(od_butterfly_sub)(tj, &tjh, tb);
  OD_KERNEL_FUNC(od_butterfly_add)(tl, &tlh, td);
  OD_KERNEL_FUNC(od_butterfly_sub)(tc, &tch, tk);
  OD_KERNEL_FUNC(od_butterfly_add)(ta, &tah, ti);

  OD_KERNEL_FUNC(od_butterfly_sub)(th, &thh, t9);
  OD_KERNEL_FUNC(od_butterfly_add)(tn, &tnh, tf);
  OD_KERNEL_FUNC(od_butterfly_sub)(te, &teh, tm);
  OD_KERNEL_FUNC(od_butterfly_add)(t8, &t8h, tg);

  OD_KERNEL_FUNC(od_butterfly_add)(t4, &t4h, t3);
  OD_KERNEL_FUNC(od_butterfly_sub)(tr, &trh, ts);
  OD_KERNEL_FUNC(od_butterfly_add)(tt, &tth, tq);
  OD_KERNEL_FUNC(od_butterfly_sub)(t2, &t2h, t5);

  OD_KERNEL_FUNC(od_butterfly_add)(t6, &t6h, t1);
  OD_KERNEL_FUNC(od_butterfly_sub)(tp, &tph, tu);
  OD_KERNEL_FUNC(od_butterfly_add)(tv, &tvh, to);
  OD_KERNEL_FUNC(od_butterfly_sub)(t0, &t0h, t7);

  /* Stage 1 */

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(t8, t8h, t7);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tn, tnh, to);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tp, tph, tm);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t6, t6h, t9);

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(ta, tah, t5);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tl, tlh, tq);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tr, trh, tk);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t4, t4h, tb);

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tc, tch, t3);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(tj, tjh, ts);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tt, tth, ti);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t2, t2h, td);

  OD_KERNEL_FUNC(od_butterfly_sub_asym)(te, teh, t1);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(th, thh, tu);
  OD_KERNEL_FUNC(od_butterfly_sub_asym)(tv, tvh, tg);
  OD_KERNEL_FUNC(od_butterfly_add_asym)(t0, t0h, tf);

  /* Stage 0 */

  /* Sin[33*Pi/128] + Cos[33*Pi/128] = 1.4137876276885337 */
  /* Sin[33*Pi/128] - Cos[33*Pi/128] = 0.0347065382144002 */
  /* Cos[33*Pi/128]                  = 0.6895405447370668 */
  od_rotate_sub(tg, tf, 16379, 14, 201, 12, 15977, 14, TX_SHIFT);

  /* Sin[35*Pi/128] + Cos[35*Pi/128] = 1.4103816894602614 */
  /* Sin[35*Pi/128] - Cos[35*Pi/128] = 0.1040360035527078 */
  /* Cos[35*Pi/128]                  = 0.6531728429537768 */
  od_rotate_add(te, th, 32679, 15, 4821, 15, 30269, 15, TX_SHIFT);

  /* Sin[37*Pi/128] + Cos[37*Pi/128] = 1.4035780182072331 */
  /* Sin[37*Pi/128] - Cos[37*Pi/128] = 0.1731148370459795 */
  /* Cos[37*Pi/128]                  = 0.6152315905806268 */
  od_rotate_sub(ti, td, 16261, 14, 4011, 14, 14255, 14, TX_SHIFT);

  /* Sin[39*Pi/128] + Cos[39*Pi/128] = 1.3933930045694290 */
  /* Sin[39*Pi/128] - Cos[39*Pi/128] = 0.2417766217337384 */
  /* Cos[39*Pi/128]                  = 0.5758081914178453 */
  od_rotate_add(tc, tj, 16143, 14, 2801, 13, 26683, 15, TX_SHIFT);

  /* Sin[41*Pi/128] + Cos[41*Pi/128] = 1.3798511851368043 */
  /* Sin[41*Pi/128] - Cos[41*Pi/128] = 0.3098559453626100 */
  /* Cos[41*Pi/128]                  = 0.5349976198870972 */
  od_rotate_sub(tk, tb, 7993, 13, 14359, 15, 3099, 12, TX_SHIFT);

  /* Sin[43*Pi/128] + Cos[43*Pi/128] = 1.3629851833384956 */
  /* Sin[43*Pi/128] - Cos[43*Pi/128] = 0.3771887988789274 */
  /* Cos[43*Pi/128]                  = 0.4928981922297840 */
  od_rotate_add(ta, tl, 31581, 15, 17479, 15, 22841, 15, TX_SHIFT);

  /* Sin[45*Pi/128] + Cos[45*Pi/128] = 1.3428356308501219 */
  /* Sin[45*Pi/128] - Cos[45*Pi/128] = 0.4436129715409088 */
  /* Cos[45*Pi/128]                  = 0.4496113296546065 */
  od_rotate_sub(tm, t9, 15557, 14, 20557, 15, 20835, 15, TX_SHIFT);

  /* Sin[47*Pi/128] + Cos[47*Pi/128] = 1.3194510697085207 */
  /* Sin[47*Pi/128] - Cos[47*Pi/128] = 0.5089684416985408 */
  /* Cos[47*Pi/128]                  = 0.4052413140049899 */
  od_rotate_add(t8, tn, 7643, 13, 11793, 14, 18779, 15, TX_SHIFT);

  /* Sin[49*Pi/128] + Cos[49*Pi/128] = 1.2928878353697270 */
  /* Sin[49*Pi/128] - Cos[49*Pi/128] = 0.5730977622997508 */
  /* Cos[49*Pi/128]                  = 0.3598950365349881 */
  od_rotate_sub(to, t7, 29957, 15, 13279, 14, 8339, 14, TX_SHIFT);

  /* Sin[51*Pi/128] + Cos[51*Pi/128] = 1.2632099209919283 */
  /* Sin[51*Pi/128] - Cos[51*Pi/128] = 0.6358464401941452 */
  /* Cos[51*Pi/128]                  = 0.3136817403988915 */
  od_rotate_add(t6, tp, 29269, 15, 14733, 14, 1817, 12, TX_SHIFT);

  /* Sin[53*Pi/128] + Cos[53*Pi/128] = 1.2304888232703382 */
  /* Sin[53*Pi/128] - Cos[53*Pi/128] = 0.6970633083205415 */
  /* Cos[53*Pi/128]                  = 0.2667127574748984 */
  od_rotate_sub(tq, t5, 28511, 15, 32303, 15, 1545, 12, TX_SHIFT);

  /* Sin[55*Pi/128] + Cos[55*Pi/128] = 1.1948033701953984 */
  /* Sin[55*Pi/128] - Cos[55*Pi/128] = 0.7566008898816587 */
  /* Cos[55*Pi/128]                  = 0.2191012401568698 */
  od_rotate_add(t4, tr, 6921, 13, 17531, 14, 10153, 15, TX_SHIFT);

  /* Sin[57*Pi/128] + Cos[57*Pi/128] = 1.1562395311492424 */
  /* Sin[57*Pi/128] - Cos[57*Pi/128] = 0.8143157536286401 */
  /* Cos[57*Pi/128]                  = 0.1709618887603012 */
  od_rotate_sub(ts, t3, 26791, 15, 4717, 12, 7923, 15, TX_SHIFT);

  /* Sin[59*Pi/128] + Cos[59*Pi/128] = 1.1148902097979262 */
  /* Sin[59*Pi/128] - Cos[59*Pi/128] = 0.8700688593994937 */
  /* Cos[59*Pi/128]                  = 0.1224106751992162 */
  od_rotate_add(t2, tt, 25833, 15, 315, 8, 5673, 15, TX_SHIFT);

  /* Sin[61*Pi/128] + Cos[61*Pi/128] = 1.0708550202783576 */
  /* Sin[61*Pi/128] - Cos[61*Pi/128] = 0.9237258930790228 */
  /* Cos[61*Pi/128]                  = 0.0735645635996674 */
  od_rotate_sub(tu, t1, 6203, 13, 21403, 14, 3409, 15, TX_SHIFT);

  /* Sin[63*Pi/128] + Cos[63*Pi/128] = 1.0242400472191164 */
  /* Sin[63*Pi/128] - Cos[63*Pi/128] = 0.9751575901732919 */
  /* Cos[63*Pi/128]                  = 0.0245412285229123 */
  od_rotate_add(t0, tv, 5933, 13, 22595, 14, 1137, 15, TX_SHIFT);
}

/* --- 64-point Transforms --- */

/**
 * 64-point orthonormal Type-II fDCT
 */
static INLINE void OD_KERNEL_FUNC(od_fdct_64)(OD_COEFF *u0, OD_COEFF *u1,
                                              OD_COEFF *u2, OD_COEFF *u3,
                                              OD_COEFF *u4, OD_COEFF *u5,
                                              OD_COEFF *u6, OD_COEFF *u7,
                                              OD_COEFF *u8, OD_COEFF *u9,
                                              OD_COEFF *ua, OD_COEFF *ub,
                                              OD_COEFF *uc, OD_COEFF *ud,
                                              OD_COEFF *ue, OD_COEFF *uf,
                                              OD_COEFF *ug, OD_COEFF *uh,
                                              OD_COEFF *ui, OD_COEFF *uj,
                                              OD_COEFF *uk, OD_COEFF *ul,
                                              OD_COEFF *um, OD_COEFF *un,
                                              OD_COEFF *uo, OD_COEFF *up,
                                              OD_COEFF *uq, OD_COEFF *ur,
                                              OD_COEFF *us, OD_COEFF *ut,
                                              OD_COEFF *uu, OD_COEFF *uv,
                                              OD_COEFF *uw, OD_COEFF *ux,
                                              OD_COEFF *uy, OD_COEFF *uz,
                                              OD_COEFF *uA, OD_COEFF *uB,
                                              OD_COEFF *uC, OD_COEFF *uD,
                                              OD_COEFF *uE, OD_COEFF *uF,
                                              OD_COEFF *uG, OD_COEFF *uH,
                                              OD_COEFF *uI, OD_COEFF *uJ,
                                              OD_COEFF *uK, OD_COEFF *uL,
                                              OD_COEFF *uM, OD_COEFF *uN,
                                              OD_COEFF *uO, OD_COEFF *uP,
                                              OD_COEFF *uQ, OD_COEFF *uR,
                                              OD_COEFF *uS, OD_COEFF *uT,
                                              OD_COEFF *uU, OD_COEFF *uV,
                                              OD_COEFF *uW, OD_COEFF *uX,
                                              OD_COEFF *uY, OD_COEFF *uZ,
                                              OD_COEFF *u_, OD_COEFF *u) {
  OD_COEFF u1h;
  OD_COEFF u3h;
  OD_COEFF u5h;
  OD_COEFF u7h;
  OD_COEFF u9h;
  OD_COEFF ubh;
  OD_COEFF udh;
  OD_COEFF ufh;
  OD_COEFF uhh;
  OD_COEFF ujh;
  OD_COEFF ulh;
  OD_COEFF unh;
  OD_COEFF uph;
  OD_COEFF urh;
  OD_COEFF uth;
  OD_COEFF uvh;
  OD_COEFF uxh;
  OD_COEFF uzh;
  OD_COEFF uBh;
  OD_COEFF uDh;
  OD_COEFF uFh;
  OD_COEFF uHh;
  OD_COEFF uJh;
  OD_COEFF uLh;
  OD_COEFF uNh;
  OD_COEFF uPh;
  OD_COEFF uRh;
  OD_COEFF uTh;
  OD_COEFF uVh;
  OD_COEFF uXh;
  OD_COEFF uZh;
  OD_COEFF uh_;

  /* +/- Butterflies with asymmetric output. */
  OD_KERNEL_FUNC(od_butterfly_neg)(u0, u , &uh_);
  OD_KERNEL_FUNC(od_butterfly_add)(u1, &u1h, u_);
  OD_KERNEL_FUNC(od_butterfly_neg)(u2, uZ, &uZh);
  OD_KERNEL_FUNC(od_butterfly_add)(u3, &u3h, uY);
  OD_KERNEL_FUNC(od_butterfly_neg)(u4, uX, &uXh);
  OD_KERNEL_FUNC(od_butterfly_add)(u5, &u5h, uW);
  OD_KERNEL_FUNC(od_butterfly_neg)(u6, uV, &uVh);
  OD_KERNEL_FUNC(od_butterfly_add)(u7, &u7h, uU);
  OD_KERNEL_FUNC(od_butterfly_neg)(u8, uT, &uTh);
  OD_KERNEL_FUNC(od_butterfly_add)(u9, &u9h, uS);
  OD_KERNEL_FUNC(od_butterfly_neg)(ua, uR, &uRh);
  OD_KERNEL_FUNC(od_butterfly_add)(ub, &ubh, uQ);
  OD_KERNEL_FUNC(od_butterfly_neg)(uc, uP, &uPh);
  OD_KERNEL_FUNC(od_butterfly_add)(ud, &udh, uO);
  OD_KERNEL_FUNC(od_butterfly_neg)(ue, uN, &uNh);
  OD_KERNEL_FUNC(od_butterfly_add)(uf, &ufh, uM);
  OD_KERNEL_FUNC(od_butterfly_neg)(ug, uL, &uLh);
  OD_KERNEL_FUNC(od_butterfly_add)(uh, &uhh, uK);
  OD_KERNEL_FUNC(od_butterfly_neg)(ui, uJ, &uJh);
  OD_KERNEL_FUNC(od_butterfly_add)(uj, &ujh, uI);
  OD_KERNEL_FUNC(od_butterfly_neg)(uk, uH, &uHh);
  OD_KERNEL_FUNC(od_butterfly_add)(ul, &ulh, uG);
  OD_KERNEL_FUNC(od_butterfly_neg)(um, uF, &uFh);
  OD_KERNEL_FUNC(od_butterfly_add)(un, &unh, uE);
  OD_KERNEL_FUNC(od_butterfly_neg)(uo, uD, &uDh);
  OD_KERNEL_FUNC(od_butterfly_add)(up, &uph, uC);
  OD_KERNEL_FUNC(od_butterfly_neg)(uq, uB, &uBh);
  OD_KERNEL_FUNC(od_butterfly_add)(ur, &urh, uA);
  OD_KERNEL_FUNC(od_butterfly_neg)(us, uz, &uzh);
  OD_KERNEL_FUNC(od_butterfly_add)(ut, &uth, uy);
  OD_KERNEL_FUNC(od_butterfly_neg)(uu, ux, &uxh);
  OD_KERNEL_FUNC(od_butterfly_add)(uv, &uvh, uw);

  /* Embedded 16-point transforms with asymmetric input. */
  OD_KERNEL_FUNC(od_fdct_32_asym)(
   u0, u1, u1h, u2, u3, u3h, u4, u5, u5h, u6, u7, u7h,
   u8, u9, u9h, ua, ub, ubh, uc, ud, udh, ue, uf, ufh,
   ug, uh, uhh, ui, uj, ujh, uk, ul, ulh, um, un, unh,
   uo, up, uph, uq, ur, urh, us, ut, uth, uu, uv, uvh);

  OD_KERNEL_FUNC(od_fdst_32_asym)(
   u , uh_, u_, uZ, uZh, uY, uX, uXh, uW, uV, uVh, uU,
   uT, uTh, uS, uR, uRh, uQ, uP, uPh, uO, uN, uNh, uM,
   uL, uLh, uK, uJ, uJh, uI, uH, uHh, uG, uF, uFh, uE,
   uD, uDh, uC, uB, uBh, uA, uz, uzh, uy, ux, uxh, uw);
}

/**
 * 64-point orthonormal Type-II iDCT
 */
static INLINE void OD_KERNEL_FUNC(od_idct_64)(OD_COEFF *u0, OD_COEFF *uw,
                                              OD_COEFF *ug, OD_COEFF *uM,
                                              OD_COEFF *u8, OD_COEFF *uE,
                                              OD_COEFF *uo, OD_COEFF *uU,
                                              OD_COEFF *u4, OD_COEFF *uA,
                                              OD_COEFF *uk, OD_COEFF *uQ,
                                              OD_COEFF *uc, OD_COEFF *uI,
                                              OD_COEFF *us, OD_COEFF *uY,
                                              OD_COEFF *u2, OD_COEFF *uy,
                                              OD_COEFF *ui, OD_COEFF *uO,
                                              OD_COEFF *ua, OD_COEFF *uG,
                                              OD_COEFF *uq, OD_COEFF *uW,
                                              OD_COEFF *u6, OD_COEFF *uC,
                                              OD_COEFF *um, OD_COEFF *uS,
                                              OD_COEFF *ue, OD_COEFF *uK,
                                              OD_COEFF *uu, OD_COEFF *u_,
                                              OD_COEFF *u1, OD_COEFF *ux,
                                              OD_COEFF *uh, OD_COEFF *uN,
                                              OD_COEFF *u9, OD_COEFF *uF,
                                              OD_COEFF *up, OD_COEFF *uV,
                                              OD_COEFF *u5, OD_COEFF *uB,
                                              OD_COEFF *ul, OD_COEFF *uR,
                                              OD_COEFF *ud, OD_COEFF *uJ,
                                              OD_COEFF *ut, OD_COEFF *uZ,
                                              OD_COEFF *u3, OD_COEFF *uz,
                                              OD_COEFF *uj, OD_COEFF *uP,
                                              OD_COEFF *ub, OD_COEFF *uH,
                                              OD_COEFF *ur, OD_COEFF *uX,
                                              OD_COEFF *u7, OD_COEFF *uD,
                                              OD_COEFF *un, OD_COEFF *uT,
                                              OD_COEFF *uf, OD_COEFF *uL,
                                              OD_COEFF *uv, OD_COEFF *u ) {
  OD_COEFF u1h;
  OD_COEFF u3h;
  OD_COEFF u5h;
  OD_COEFF u7h;
  OD_COEFF u9h;
  OD_COEFF ubh;
  OD_COEFF udh;
  OD_COEFF ufh;
  OD_COEFF uhh;
  OD_COEFF ujh;
  OD_COEFF ulh;
  OD_COEFF unh;
  OD_COEFF uph;
  OD_COEFF urh;
  OD_COEFF uth;
  OD_COEFF uvh;

  /* Embedded 32-point transforms with asymmetric output. */
  OD_KERNEL_FUNC(od_idst_32_asym)(
   u , uL, uT, uD, uX, uH, uP, uz, uZ, uJ, uR, uB, uV, uF, uN, ux,
   u_, uK, uS, uC, uW, uG, uO, uy, uY, uI, uQ, uA, uU, uE, uM, uw);
  OD_KERNEL_FUNC(od_idct_32_asym)(
   u0, ug, u8, uo, u4, uk, uc, us, u2, ui, ua, uq, u6, um, ue, uu,
   u1, &u1h, uh, &uhh, u9, &u9h, up, &uph,
   u5, &u5h, ul, &ulh, ud, &udh, ut, &uth,
   u3, &u3h, uj, &ujh, ub, &ubh, ur, &urh,
   u7, &u7h, un, &unh, uf, &ufh, uv, &uvh);

  /* +/- Butterflies with asymmetric input. */
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(u0, u , OD_RSHIFT1(*u));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(u1, u1h, u_);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(u2, uZ, OD_RSHIFT1(*uZ));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(u3, u3h, uY);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(u4, uX, OD_RSHIFT1(*uX));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(u5, u5h, uW);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(u6, uV, OD_RSHIFT1(*uV));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(u7, u7h, uU);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(u8, uT, OD_RSHIFT1(*uT));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(u9, u9h, uS);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(ua, uR, OD_RSHIFT1(*uR));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(ub, ubh, uQ);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(uc, uP, OD_RSHIFT1(*uP));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(ud, udh, uO);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(ue, uN, OD_RSHIFT1(*uN));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(uf, ufh, uM);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(ug, uL, OD_RSHIFT1(*uL));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(uh, uhh, uK);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(ui, uJ, OD_RSHIFT1(*uJ));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(uj, ujh, uI);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(uk, uH, OD_RSHIFT1(*uH));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(ul, ulh, uG);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(um, uF, OD_RSHIFT1(*uF));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(un, unh, uE);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(uo, uD, OD_RSHIFT1(*uD));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(up, uph, uC);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(uq, uB, OD_RSHIFT1(*uB));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(ur, urh, uA);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(us, uz, OD_RSHIFT1(*uz));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(ut, uth, uy);
  OD_KERNEL_FUNC(od_butterfly_neg_asym)(uu, ux, OD_RSHIFT1(*ux));
  OD_KERNEL_FUNC(od_butterfly_add_asym)(uv, uvh, uw);
}
