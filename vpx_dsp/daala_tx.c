#include "vpx_dsp/daala_tx.h"
#include "vpx_dsp/odintrin.h"

#define CONFIG_DAALA_TX_DST8 1
#define CONFIG_DAALA_TX_DST32 0
#define CONFIG_TX64X64 0

static INLINE od_coeff od_add(od_coeff p0, od_coeff p1) { return p0 + p1; }

static INLINE od_coeff od_sub(od_coeff p0, od_coeff p1) { return p0 - p1; }

static INLINE od_coeff od_add_avg(od_coeff p0, od_coeff p1) {
  return (od_add(p0, p1) + TX_AVG_BIAS) >> 1;
}

static INLINE od_coeff od_sub_avg(od_coeff p0, od_coeff p1) {
  return (od_sub(p0, p1) + TX_AVG_BIAS) >> 1;
}

static INLINE od_coeff od_rshift1(od_coeff v) { return (v + (v < 0)) >> 1; }

/* Fixed point multiply. */
static INLINE od_coeff od_mul(od_coeff n, int c, int q) {
  //return (n * c + ((1 << q) >> 1)) >> q;
  return (((int)n) * c + ((1 << q) >> 1)) >> q;
}

/* Swap coefficient values. */
static INLINE void od_swap(od_coeff *q0, od_coeff *q1) {
  od_coeff t;
  t = *q0;
  *q0 = *q1;
  *q1 = t;
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
#define OD_KERNEL c
#define OD_COEFF od_coeff
#define OD_ADD od_add
#define OD_SUB od_sub
#define OD_RSHIFT1 od_rshift1
#define OD_ADD_AVG od_add_avg
#define OD_SUB_AVG od_sub_avg
#define OD_MUL od_mul
#define OD_SWAP od_swap

#include "vpx_dsp/daala_tx_kernels.h"

/* clang-format off */

/* 4-point orthonormal Type-II fDCT. */
void od_bin_fdct4(od_coeff y[4], const od_coeff *x, int xstride) {
  int q0;
  int q1;
  int q2;
  int q3;
  q0 = x[0*xstride];
  q1 = x[1*xstride];
  q2 = x[2*xstride];
  q3 = x[3*xstride];
  od_fdct_4_c(&q0, &q1, &q2, &q3);
  y[0] = (od_coeff)q0;
  y[1] = (od_coeff)q2;
  y[2] = (od_coeff)q1;
  y[3] = (od_coeff)q3;
}

/* 4-point orthonormal Type-VII fDST. */
void od_bin_fdst4(od_coeff y[4], const od_coeff *x, int xstride) {
  int q0;
  int q1;
  int q2;
  int q3;
  q0 = x[0*xstride];
  q1 = x[1*xstride];
  q2 = x[2*xstride];
  q3 = x[3*xstride];
  od_fdst_vii_4_c(&q0, &q1, &q2, &q3);
  y[0] = (od_coeff)q0;
  y[1] = (od_coeff)q2;
  y[2] = (od_coeff)q1;
  y[3] = (od_coeff)q3;
}

void od_bin_fdct8(od_coeff y[8], const od_coeff *x, int xstride) {
  int r0;
  int r1;
  int r2;
  int r3;
  int r4;
  int r5;
  int r6;
  int r7;
  r0 = x[0*xstride];
  r1 = x[1*xstride];
  r2 = x[2*xstride];
  r3 = x[3*xstride];
  r4 = x[4*xstride];
  r5 = x[5*xstride];
  r6 = x[6*xstride];
  r7 = x[7*xstride];
  od_fdct_8_c(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
  y[0] = (od_coeff)r0;
  y[1] = (od_coeff)r4;
  y[2] = (od_coeff)r2;
  y[3] = (od_coeff)r6;
  y[4] = (od_coeff)r1;
  y[5] = (od_coeff)r5;
  y[6] = (od_coeff)r3;
  y[7] = (od_coeff)r7;
}

#if !CONFIG_DAALA_TX_DST8

#define OD_PAVG(_a, _b) (((_a) + (_b) + 1) >> 1)

void od_bin_fdst8(od_coeff y[8], const od_coeff *x, int xstride) {
  int r0;
  int r1;
  int r2;
  int r3;
  int r4;
  int r5;
  int r6;
  int r7;
  r0 = x[0*xstride];
  r1 = x[1*xstride];
  r2 = x[2*xstride];
  r3 = x[3*xstride];
  r4 = x[4*xstride];
  r5 = x[5*xstride];
  r6 = x[6*xstride];
  r7 = x[7*xstride];
  od_fdst_8_c(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
  y[0] = (od_coeff)r0;
  y[1] = (od_coeff)r4;
  y[2] = (od_coeff)r2;
  y[3] = (od_coeff)r6;
  y[4] = (od_coeff)r1;
  y[5] = (od_coeff)r5;
  y[6] = (od_coeff)r3;
  y[7] = (od_coeff)r7;
}
#else
const int OD_DST_8_PERM[8] = { 0, 7, 1, 6, 2, 5, 3, 4 };

/* Computes the Polynomial Product Y(z) ≡ X(z)*H(z) modulo (z^8 + 1) using
    Nussbaumer's "short" algorithm [1].
   The monomial coefficients in Y(z) are exactly the values of an acyclic
    convolution of the monomial coefficients of X(z) and H(z).
   Since H(z) is fixed, the multiplication terms are constant and precomputed.

   [1] Nussbaumer, Henri J. "Fast Fourier Transform and Convolution Algorithms"
        Springer-Verlag: Berlin, Heidelberg, New York (1981) pages 76-78. */
static void od_poly_prod_8(od_coeff y[8], const od_coeff x[8]) {
  /* 21 "muls", 76 adds, 21 shifts */
  od_coeff q0;
  od_coeff q1;
  od_coeff q2;
  od_coeff q3;
  od_coeff q4;
  od_coeff q5;
  od_coeff q6;
  od_coeff q7;
  od_coeff q8;
  od_coeff q9;
  od_coeff q10;
  od_coeff q11;
  od_coeff q12;
  od_coeff q13;
  od_coeff q14;
  od_coeff q15;
  od_coeff q16;
  od_coeff q17;
  od_coeff q18;
  od_coeff q19;
  od_coeff q20;
  od_coeff r0;
  od_coeff r1;
  od_coeff r2;
  od_coeff r3;
  od_coeff r4;
  od_coeff r5;
  od_coeff r6;
  od_coeff r7;
  od_coeff t0;
  od_coeff t1;
  od_coeff t2;
  od_coeff t3;
  od_coeff t4;
  od_coeff t5;
  od_coeff t6;
  od_coeff t7;
  od_coeff u0;
  od_coeff u1;
  od_coeff u1h;
  od_coeff u2;
  od_coeff u2h;
  od_coeff u3;
  od_coeff u4;
  od_coeff u4h;
  od_coeff u5;
  od_coeff u6;
  od_coeff u7;
  od_coeff u7h;
  od_coeff u8;
  od_coeff u9;
  od_coeff u10;
  od_coeff u11;
  od_coeff u12;
  od_coeff u13;
  od_coeff u14;
  od_coeff u15;
  od_coeff u16;
  od_coeff u17;
  od_coeff u18;
  od_coeff u19;
  od_coeff u20;
  od_coeff u21;
  od_coeff u22;
  od_coeff u23;
  od_coeff u24;
  od_coeff u25;
  od_coeff u26;
  od_coeff u27;
  t0 = x[0];
  t1 = x[1];
  t2 = x[2];
  t3 = x[3];
  t4 = x[4];
  t5 = x[5];
  t6 = x[6];
  t7 = x[7];
  /* Stage 0 Butterfly */
  u7 = t0 - t7;
  u7h = OD_RSHIFT1(u7);
  u0 = t0 - u7h;
  u2 = t2 - t6;
  u2h = OD_RSHIFT1(u2);
  u6 = t2 - u2h;
  u4 = t4 + t5;
  u4h = OD_RSHIFT1(u4);
  u5 = t4 - u4h;
  u1 = t3 - t1;
  u1h = OD_RSHIFT1(u1);
  u3 = t3 - u1h;
  /* Stage 1 Butterfly */
  q0 = u0 + u2h;
  q1 = q0 - u2;
  q4 = u3 + u4h;
  q5 = q4 - u4;
  q2 = u7h + u5;
  q7 = u7 - q2;
  q6 = u1h + u6;
  q3 = u1 - q6;
  /* Stage 2 Half-Butterfly */
  /*The intermediate sums can overflow 16 bits, but all SIMD instruction sets
     should be able to compute them without issue (i.e., using PAVGW or
     V{R}HADD.S16).*/
  q8 = (q0 + q4 + 1) >> 1;
  q9 = (q1 + q5) >> 1;
  q10 = (q2 + q3 + 1) >> 1;
  q11 = (q7 + q6) >> 1;
  /* Stage 3 */
  q12 = t0 + t3;
  q13 = t0;
  q14 = t3;
  q15 = t5 - t6;
  q16 = t6;
  q17 = t5;
  r0 = t2 + t4;
  r1 = t2 - OD_RSHIFT1(r0);
  r2 = (r1 - q15 + 1) >> 1;
  r3 = OD_RSHIFT1(t0);
  r4 = (r3 - t1 + 1) >> 1;
  /* q18 = (q6 - q4)/2 + (t0 - q15)/4
         = (t0 + t2 - t4)/4 - (t1 + t5 - t6)/2 */
  q18 = r2 + r4;
  r5 = t5 - (q15 >> 1);
  r6 = (r0 + t3 + 1) >> 1;
  r7 = (t7 + r6 + 1) >> 1;
  /* q19 = (q7 - q0)/2 + (t5 + t6 - t3)/4
         = (t5 + t6 - t7)/2 - (t2 + t3 + t4)/4 */
  q19 = r5 - r7;
  q20 = (q18 - q19) >> 1;
  /* Stage 4 */
  q0 = (-5995*q0 + 8192) >> 14;
  q1 = (-1373*q1 + 4096) >> 13;
  q2 = (22891*q2 + 16384) >> 15;
  q3 = (-217*q3 + 512) >> 10;
  q4 = (13427*q4 + 16384) >> 15;
  q5 = (-11013*q5 + 8192) >> 14;
  q6 = (1373*q6 + 1024) >> 11;
  q7 = (-14077*q7 + 16384) >> 15;
  q8 = (-1437*q8 + 16384) >> 15;
  q9 = (27519*q9 + 16384) >> 15;
  q10 = (-15947*q10 + 16384) >> 15;
  q11 = (-7891*q11 + 16384) >> 15;
  q12 = (4897*q12 + 16384) >> 15;
  q13 = (-5079*q13 + 8192) >> 14;
  q14 = (365*q14 + 16384) >> 15;
  q15 = (3325*q15 + 8192) >> 14;
  q16 = (-5225*q16 + 8192) >> 14;
  q17 = (-1425*q17 + 8192) >> 14;
  q18 = (3453*q18 + 16384) >> 15;
  q19 = (-8421*q19 + 8192) >> 14;
  q20 = (-20295*q20 + 16384) >> 15;
  /* Stage 5 */
  u0 = q0 + q8;
  u1 = q1 + q9;
  u2 = q2 + q10;
  u3 = q3 + q10;
  u4 = q4 + q8;
  u5 = q5 + q9;
  u6 = q6 + q11;
  u7 = q7 + q11;
  /* Stage 6 */
  u10 = u0 + u1;
  u11 = u0 - u1;
  u12 = u2 + u7;
  u13 = u2 - u7;
  u14 = u3 + u6;
  u15 = u3 - u6;
  u16 = u5 + u4;
  u17 = u5 - u4;
  /* Stage 7 */
  u8 = q19 + q20;
  u9 = q19 - q18;
  u18 = q12 + u8;
  u19 = u18 + q13;
  u20 = u18 + q14;
  u21 = 2*u9;
  u22 = q15 + u21;
  u23 = q16 - u22;
  u24 = u22 + q17;
  u25 = 2*u8;
  u26 = 2*u25;
  u27 = u25 - u9;
  /* Stage 8 */
  y[0] = u14 + u16 + u20;
  y[1] = u12 - u10 - u25;
  y[2] = u9 + u13 - u17;
  y[3] = u9 - u10 - u12 - u19;
  y[4] = u15 - u11 - u27;
  y[5] = u23 - u11 - u15;
  y[6] = u13 + u17 - u24 + u26;
  y[7] = u16 - u14 + u21 - u25;
}

void od_bin_fdst8(od_coeff y[8], const od_coeff *x, int xstride) {
  int i;
  od_coeff xp[8];
  od_coeff yp[8];
  for (i = 0; i < 8; i++) xp[i] = x[i*xstride];
  od_poly_prod_8(yp, xp);
  for (i = 0; i < 8; i++) y[OD_DST_8_PERM[i]] = yp[i];
}
#endif
void od_bin_fdct16(od_coeff y[16], const od_coeff *x, int xstride) {
  int s0;
  int s1;
  int s2;
  int s3;
  int s4;
  int s5;
  int s6;
  int s7;
  int s8;
  int s9;
  int sa;
  int sb;
  int sc;
  int sd;
  int se;
  int sf;
  s0 = x[0*xstride];
  s1 = x[1*xstride];
  s2 = x[2*xstride];
  s3 = x[3*xstride];
  s4 = x[4*xstride];
  s5 = x[5*xstride];
  s6 = x[6*xstride];
  s7 = x[7*xstride];
  s8 = x[8*xstride];
  s9 = x[9*xstride];
  sa = x[10*xstride];
  sb = x[11*xstride];
  sc = x[12*xstride];
  sd = x[13*xstride];
  se = x[14*xstride];
  sf = x[15*xstride];
  od_fdct_16_c(&s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7,
    &s8, &s9, &sa, &sb, &sc, &sd, &se, &sf);
  y[0] = (od_coeff)s0;
  y[1] = (od_coeff)s8;
  y[2] = (od_coeff)s4;
  y[3] = (od_coeff)sc;
  y[4] = (od_coeff)s2;
  y[5] = (od_coeff)sa;
  y[6] = (od_coeff)s6;
  y[7] = (od_coeff)se;
  y[8] = (od_coeff)s1;
  y[9] = (od_coeff)s9;
  y[10] = (od_coeff)s5;
  y[11] = (od_coeff)sd;
  y[12] = (od_coeff)s3;
  y[13] = (od_coeff)sb;
  y[14] = (od_coeff)s7;
  y[15] = (od_coeff)sf;
}

void od_bin_fdst16(od_coeff y[16], const od_coeff *x, int xstride) {
  int s0;
  int s1;
  int s2;
  int s3;
  int s4;
  int s5;
  int s6;
  int s7;
  int s8;
  int s9;
  int sa;
  int sb;
  int sc;
  int sd;
  int se;
  int sf;
  s0 = x[0*xstride];
  s1 = x[1*xstride];
  s2 = x[2*xstride];
  s3 = x[3*xstride];
  s4 = x[4*xstride];
  s5 = x[5*xstride];
  s6 = x[6*xstride];
  s7 = x[7*xstride];
  s8 = x[8*xstride];
  s9 = x[9*xstride];
  sa = x[10*xstride];
  sb = x[11*xstride];
  sc = x[12*xstride];
  sd = x[13*xstride];
  se = x[14*xstride];
  sf = x[15*xstride];
  od_fdst_16_c(&s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7,
    &s8, &s9, &sa, &sb, &sc, &sd, &se, &sf);
  y[0] = (od_coeff)s0;
  y[1] = (od_coeff)s8;
  y[2] = (od_coeff)s4;
  y[3] = (od_coeff)sc;
  y[4] = (od_coeff)s2;
  y[5] = (od_coeff)sa;
  y[6] = (od_coeff)s6;
  y[7] = (od_coeff)se;
  y[8] = (od_coeff)s1;
  y[9] = (od_coeff)s9;
  y[10] = (od_coeff)s5;
  y[11] = (od_coeff)sd;
  y[12] = (od_coeff)s3;
  y[13] = (od_coeff)sb;
  y[14] = (od_coeff)s7;
  y[15] = (od_coeff)sf;
}

void od_bin_fdct32(od_coeff y[32], const od_coeff *x, int xstride) {
  /*215 adds, 38 shifts, 87 "muls".*/
  int t0;
  int t1;
  int t2;
  int t3;
  int t4;
  int t5;
  int t6;
  int t7;
  int t8;
  int t9;
  int ta;
  int tb;
  int tc;
  int td;
  int te;
  int tf;
  int tg;
  int th;
  int ti;
  int tj;
  int tk;
  int tl;
  int tm;
  int tn;
  int to;
  int tp;
  int tq;
  int tr;
  int ts;
  int tt;
  int tu;
  int tv;
  t0 = x[0*xstride];
  tg = x[1*xstride];
  t8 = x[2*xstride];
  to = x[3*xstride];
  t4 = x[4*xstride];
  tk = x[5*xstride];
  tc = x[6*xstride];
  ts = x[7*xstride];
  t2 = x[8*xstride];
  ti = x[9*xstride];
  ta = x[10*xstride];
  tq = x[11*xstride];
  t6 = x[12*xstride];
  tm = x[13*xstride];
  te = x[14*xstride];
  tu = x[15*xstride];
  t1 = x[16*xstride];
  th = x[17*xstride];
  t9 = x[18*xstride];
  tp = x[19*xstride];
  t5 = x[20*xstride];
  tl = x[21*xstride];
  td = x[22*xstride];
  tt = x[23*xstride];
  t3 = x[24*xstride];
  tj = x[25*xstride];
  tb = x[26*xstride];
  tr = x[27*xstride];
  t7 = x[28*xstride];
  tn = x[29*xstride];
  tf = x[30*xstride];
  tv = x[31*xstride];
  od_fdct_32_c(
    &t0, &tg, &t8, &to, &t4, &tk, &tc, &ts, &t2, &ti, &ta, &tq, &t6, &tm, &te,
    &tu, &t1, &th, &t9, &tp, &t5, &tl, &td, &tt, &t3, &tj, &tb, &tr, &t7, &tn,
    &tf, &tv);
  y[0] = (od_coeff)t0;
  y[1] = (od_coeff)t1;
  y[2] = (od_coeff)t2;
  y[3] = (od_coeff)t3;
  y[4] = (od_coeff)t4;
  y[5] = (od_coeff)t5;
  y[6] = (od_coeff)t6;
  y[7] = (od_coeff)t7;
  y[8] = (od_coeff)t8;
  y[9] = (od_coeff)t9;
  y[10] = (od_coeff)ta;
  y[11] = (od_coeff)tb;
  y[12] = (od_coeff)tc;
  y[13] = (od_coeff)td;
  y[14] = (od_coeff)te;
  y[15] = (od_coeff)tf;
  y[16] = (od_coeff)tg;
  y[17] = (od_coeff)th;
  y[18] = (od_coeff)ti;
  y[19] = (od_coeff)tj;
  y[20] = (od_coeff)tk;
  y[21] = (od_coeff)tl;
  y[22] = (od_coeff)tm;
  y[23] = (od_coeff)tn;
  y[24] = (od_coeff)to;
  y[25] = (od_coeff)tp;
  y[26] = (od_coeff)tq;
  y[27] = (od_coeff)tr;
  y[28] = (od_coeff)ts;
  y[29] = (od_coeff)tt;
  y[30] = (od_coeff)tu;
  y[31] = (od_coeff)tv;
}

void od_bin_fdst32(od_coeff y[32], const od_coeff *x, int xstride) {
  od_coeff t0;
  od_coeff t1;
  od_coeff t2;
  od_coeff t3;
  od_coeff t4;
  od_coeff t5;
  od_coeff t6;
  od_coeff t7;
  od_coeff t8;
  od_coeff t9;
  od_coeff ta;
  od_coeff tb;
  od_coeff tc;
  od_coeff td;
  od_coeff te;
  od_coeff tf;
  od_coeff tg;
  od_coeff th;
  od_coeff ti;
  od_coeff tj;
  od_coeff tk;
  od_coeff tl;
  od_coeff tm;
  od_coeff tn;
  od_coeff to;
  od_coeff tp;
  od_coeff tq;
  od_coeff tr;
  od_coeff ts;
  od_coeff tt;
  od_coeff tu;
  od_coeff tv;
  #if !CONFIG_DAALA_TX_DST32
    assert(0 && "od_bin_fdst32() called when !CONFIG_DAALA_TX_DST32");
  #endif
  t0 = x[0*xstride];
  t1 = x[1*xstride];
  t2 = x[2*xstride];
  t3 = x[3*xstride];
  t4 = x[4*xstride];
  t5 = x[5*xstride];
  t6 = x[6*xstride];
  t7 = x[7*xstride];
  t8 = x[8*xstride];
  t9 = x[9*xstride];
  ta = x[10*xstride];
  tb = x[11*xstride];
  tc = x[12*xstride];
  td = x[13*xstride];
  te = x[14*xstride];
  tf = x[15*xstride];
  tg = x[16*xstride];
  th = x[17*xstride];
  ti = x[18*xstride];
  tj = x[19*xstride];
  tk = x[20*xstride];
  tl = x[21*xstride];
  tm = x[22*xstride];
  tn = x[23*xstride];
  to = x[24*xstride];
  tp = x[25*xstride];
  tq = x[26*xstride];
  tr = x[27*xstride];
  ts = x[28*xstride];
  tt = x[29*xstride];
  tu = x[30*xstride];
  tv = x[31*xstride];
  od_fdst_32_c(
    &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9, &ta, &tb, &tc, &td, &te,
    &tf, &tg, &th, &ti, &tj, &tk, &tl, &tm, &tn, &to, &tp, &tq, &tr, &ts, &tt,
    &tu, &tv);
  y[0] = t0;
  y[1] = tg;
  y[2] = t8;
  y[3] = to;
  y[4] = t4;
  y[5] = tk;
  y[6] = tc;
  y[7] = ts;
  y[8] = t2;
  y[9] = ti;
  y[10] = ta;
  y[11] = tq;
  y[12] = t6;
  y[13] = tm;
  y[14] = te;
  y[15] = tu;
  y[16] = t1;
  y[17] = th;
  y[18] = t9;
  y[19] = tp;
  y[20] = t5;
  y[21] = tl;
  y[22] = td;
  y[23] = tt;
  y[24] = t3;
  y[25] = tj;
  y[26] = tb;
  y[27] = tr;
  y[28] = t7;
  y[29] = tn;
  y[30] = tf;
  y[31] = tv;
}

#if CONFIG_TX64X64
void od_bin_fdct64(od_coeff y[64], const od_coeff *x, int xstride) {
  int u0;
  int u1;
  int u2;
  int u3;
  int u4;
  int u5;
  int u6;
  int u7;
  int u8;
  int u9;
  int ua;
  int ub;
  int uc;
  int ud;
  int ue;
  int uf;
  int ug;
  int uh;
  int ui;
  int uj;
  int uk;
  int ul;
  int um;
  int un;
  int uo;
  int up;
  int uq;
  int ur;
  int us;
  int ut;
  int uu;
  int uv;
  int uw;
  int ux;
  int uy;
  int uz;
  int uA;
  int uB;
  int uC;
  int uD;
  int uE;
  int uF;
  int uG;
  int uH;
  int uI;
  int uJ;
  int uK;
  int uL;
  int uM;
  int uN;
  int uO;
  int uP;
  int uQ;
  int uR;
  int uS;
  int uT;
  int uU;
  int uV;
  int uW;
  int uX;
  int uY;
  int uZ;
  int u_;
  int u;
  u0 = x[0*xstride];
  uw = x[1*xstride];
  ug = x[2*xstride];
  uM = x[3*xstride];
  u8 = x[4*xstride];
  uE = x[5*xstride];
  uo = x[6*xstride];
  uU = x[7*xstride];
  u4 = x[8*xstride];
  uA = x[9*xstride];
  uk = x[10*xstride];
  uQ = x[11*xstride];
  uc = x[12*xstride];
  uI = x[13*xstride];
  us = x[14*xstride];
  uY = x[15*xstride];
  u2 = x[16*xstride];
  uy = x[17*xstride];
  ui = x[18*xstride];
  uO = x[19*xstride];
  ua = x[20*xstride];
  uG = x[21*xstride];
  uq = x[22*xstride];
  uW = x[23*xstride];
  u6 = x[24*xstride];
  uC = x[25*xstride];
  um = x[26*xstride];
  uS = x[27*xstride];
  ue = x[28*xstride];
  uK = x[29*xstride];
  uu = x[30*xstride];
  u_ = x[31*xstride];
  u1 = x[32*xstride];
  ux = x[33*xstride];
  uh = x[34*xstride];
  uN = x[35*xstride];
  u9 = x[36*xstride];
  uF = x[37*xstride];
  up = x[38*xstride];
  uV = x[39*xstride];
  u5 = x[40*xstride];
  uB = x[41*xstride];
  ul = x[42*xstride];
  uR = x[43*xstride];
  ud = x[44*xstride];
  uJ = x[45*xstride];
  ut = x[46*xstride];
  uZ = x[47*xstride];
  u3 = x[48*xstride];
  uz = x[49*xstride];
  uj = x[50*xstride];
  uP = x[51*xstride];
  ub = x[52*xstride];
  uH = x[53*xstride];
  ur = x[54*xstride];
  uX = x[55*xstride];
  u7 = x[56*xstride];
  uD = x[57*xstride];
  un = x[58*xstride];
  uT = x[59*xstride];
  uf = x[60*xstride];
  uL = x[61*xstride];
  uv = x[62*xstride];
  u  = x[63*xstride];
  od_fdct_64_c(
    &u0, &uw, &ug, &uM, &u8, &uE, &uo, &uU, &u4, &uA, &uk, &uQ, &uc, &uI, &us,
    &uY, &u2, &uy, &ui, &uO, &ua, &uG, &uq, &uW, &u6, &uC, &um, &uS, &ue, &uK,
    &uu, &u_, &u1, &ux, &uh, &uN, &u9, &uF, &up, &uV, &u5, &uB, &ul, &uR, &ud,
    &uJ, &ut, &uZ, &u3, &uz, &uj, &uP, &ub, &uH, &ur, &uX, &u7, &uD, &un, &uT,
    &uf, &uL, &uv, &u );
  y[0] = (od_coeff)u0;
  y[1] = (od_coeff)u1;
  y[2] = (od_coeff)u2;
  y[3] = (od_coeff)u3;
  y[4] = (od_coeff)u4;
  y[5] = (od_coeff)u5;
  y[6] = (od_coeff)u6;
  y[7] = (od_coeff)u7;
  y[8] = (od_coeff)u8;
  y[9] = (od_coeff)u9;
  y[10] = (od_coeff)ua;
  y[11] = (od_coeff)ub;
  y[12] = (od_coeff)uc;
  y[13] = (od_coeff)ud;
  y[14] = (od_coeff)ue;
  y[15] = (od_coeff)uf;
  y[16] = (od_coeff)ug;
  y[17] = (od_coeff)uh;
  y[18] = (od_coeff)ui;
  y[19] = (od_coeff)uj;
  y[20] = (od_coeff)uk;
  y[21] = (od_coeff)ul;
  y[22] = (od_coeff)um;
  y[23] = (od_coeff)un;
  y[24] = (od_coeff)uo;
  y[25] = (od_coeff)up;
  y[26] = (od_coeff)uq;
  y[27] = (od_coeff)ur;
  y[28] = (od_coeff)us;
  y[29] = (od_coeff)ut;
  y[30] = (od_coeff)uu;
  y[31] = (od_coeff)uv;
  y[32] = (od_coeff)uw;
  y[33] = (od_coeff)ux;
  y[34] = (od_coeff)uy;
  y[35] = (od_coeff)uz;
  y[36] = (od_coeff)uA;
  y[37] = (od_coeff)uB;
  y[38] = (od_coeff)uC;
  y[39] = (od_coeff)uD;
  y[40] = (od_coeff)uE;
  y[41] = (od_coeff)uF;
  y[41] = (od_coeff)uF;
  y[42] = (od_coeff)uG;
  y[43] = (od_coeff)uH;
  y[44] = (od_coeff)uI;
  y[45] = (od_coeff)uJ;
  y[46] = (od_coeff)uK;
  y[47] = (od_coeff)uL;
  y[48] = (od_coeff)uM;
  y[49] = (od_coeff)uN;
  y[50] = (od_coeff)uO;
  y[51] = (od_coeff)uP;
  y[52] = (od_coeff)uQ;
  y[53] = (od_coeff)uR;
  y[54] = (od_coeff)uS;
  y[55] = (od_coeff)uT;
  y[56] = (od_coeff)uU;
  y[57] = (od_coeff)uV;
  y[58] = (od_coeff)uW;
  y[59] = (od_coeff)uX;
  y[60] = (od_coeff)uY;
  y[61] = (od_coeff)uZ;
  y[62] = (od_coeff)u_;
  y[63] = (od_coeff)u;
}
#endif

void od_bin_fidtx4(od_coeff y[4], const od_coeff *x, int xstride) {
  int i;
  for (i = 0; i < 4; i++)
    y[i] = x[i*xstride];
}

void od_bin_fidtx8(od_coeff y[8], const od_coeff *x, int xstride) {
  int i;
  for (i = 0; i < 8; i++)
    y[i] = x[i*xstride];
}

void od_bin_fidtx16(od_coeff y[16], const od_coeff *x, int xstride) {
  int i;
  for (i = 0; i < 16; i++)
    y[i] = x[i*xstride];
}

void od_bin_fidtx32(od_coeff y[32], const od_coeff *x, int xstride) {
  int i;
  for (i = 0; i < 32; i++)
    y[i] = x[i*xstride];
}

#if CONFIG_TX64X64
void od_bin_fidtx64(od_coeff y[64], const od_coeff *x, int xstride) {
  int i;
  for (i = 0; i < 64; i++)
    y[i] = x[i*xstride];
}
#endif

// Below are intermediate wrappers that handle the case when
// tran_low_t is a smaller type than od_coeff
void daala_fdct4(const tran_low_t *input, tran_low_t *output) {
  int i;
  od_coeff x[4];
  od_coeff y[4];
  for (i = 0; i < 4; i++) x[i] = (od_coeff)input[i];
  od_bin_fdct4(y, x, 1);
  for (i = 0; i < 4; i++) output[i] = (tran_low_t)y[i];
}

void daala_fdst4(const tran_low_t *input, tran_low_t *output) {
  int i;
  od_coeff x[4];
  od_coeff y[4];
  for (i = 0; i < 4; i++) x[i] = (od_coeff)input[i];
  od_bin_fdst4(y, x, 1);
  for (i = 0; i < 4; i++) output[i] = (tran_low_t)y[i];
}

void daala_fdct8(const tran_low_t *input, tran_low_t *output) {
  int i;
  od_coeff x[8];
  od_coeff y[8];
  for (i = 0; i < 8; i++) x[i] = (od_coeff)input[i];
  od_bin_fdct8(y, x, 1);
  for (i = 0; i < 8; i++) output[i] = (tran_low_t)y[i];
}

void daala_fdst8(const tran_low_t *input, tran_low_t *output) {
  int i;
  od_coeff x[8];
  od_coeff y[8];
  for (i = 0; i < 8; i++) x[i] = (od_coeff)input[i];
  od_bin_fdst8(y, x, 1);
  for (i = 0; i < 8; i++) output[i] = (tran_low_t)y[i];
}

void daala_fdct16(const tran_low_t *input, tran_low_t *output) {
  int i;
  od_coeff x[16];
  od_coeff y[16];
  for (i = 0; i < 16; i++) x[i] = (od_coeff)input[i];
  od_bin_fdct16(y, x, 1);
  for (i = 0; i < 16; i++) output[i] = (tran_low_t)y[i];
}

void daala_fdst16(const tran_low_t *input, tran_low_t *output) {
  int i;
  od_coeff x[16];
  od_coeff y[16];
  for (i = 0; i < 16; i++) x[i] = (od_coeff)input[i];
  od_bin_fdst16(y, x, 1);
  for (i = 0; i < 16; i++) output[i] = (tran_low_t)y[i];
}

void daala_fdct32(const tran_low_t *input, tran_low_t *output) {
  int i;
  od_coeff x[32];
  od_coeff y[32];
  for (i = 0; i < 32; i++) x[i] = (od_coeff)input[i];
  od_bin_fdct32(y, x, 1);
  for (i = 0; i < 32; i++) output[i] = (tran_low_t)y[i];
}

void daala_fdst32(const tran_low_t *input, tran_low_t *output) {
  int i;
  od_coeff x[32];
  od_coeff y[32];
  for (i = 0; i < 32; i++) x[i] = (od_coeff)input[i];
  od_bin_fdst32(y, x, 1);
  for (i = 0; i < 32; i++) output[i] = (tran_low_t)y[i];
}

#if CONFIG_TX64X64
void daala_fdct64(const tran_low_t *input, tran_low_t *output) {
  int i;
  od_coeff x[64];
  od_coeff y[64];
  for (i = 0; i < 64; i++) x[i] = (od_coeff)input[i];
  od_bin_fdct64(y, x, 1);
  for (i = 0; i < 64; i++) output[i] = (tran_low_t)y[i];
}

/* Preserve the "half-right" transform behavior. */
void daala_fdst64(const tran_low_t *input, tran_low_t *output) {
  int i;
  tran_low_t inputhalf[32];
  for (i = 0; i < 32; ++i) {
    output[32 + i] = input[i];
  }
  for (i = 0; i < 32; ++i) {
    inputhalf[i] = input[i + 32];
  }
  daala_fdct32(inputhalf, output);
}

void daala_idtx64(const tran_low_t *input, tran_low_t *output) {
  int i;
  for (i = 0; i < 64; i++) output[i] = input[i];
}
#endif
