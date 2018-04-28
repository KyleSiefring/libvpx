#ifndef AOM_DSP_DAALA_TX_H_
#define AOM_DSP_DAALA_TX_H_

#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_dsp/odintrin.h"

/*Controls the biasing of average operations.
  Must be either 0 or 1.*/
#define TX_AVG_BIAS (0)

/*Parameterizes the variations of the transform kernels.
  See daala_tx_kernels.h for usage.*/
#define TX_NONE (0)
#define TX_AVG (!TX_NONE)
#define TX_SHIFT (!TX_NONE)
#define TX_ADD (0)
#define TX_SUB (1)

void daala_fdct4(const tran_low_t *input, tran_low_t *output);
void daala_idct4(const tran_low_t *input, tran_low_t *output);
void daala_fdst4(const tran_low_t *input, tran_low_t *output);
void daala_idst4(const tran_low_t *input, tran_low_t *output);
void daala_idtx4(const tran_low_t *input, tran_low_t *output);
void daala_fdct8(const tran_low_t *input, tran_low_t *output);
void daala_idct8(const tran_low_t *input, tran_low_t *output);
void daala_fdst8(const tran_low_t *input, tran_low_t *output);
void daala_idst8(const tran_low_t *input, tran_low_t *output);
void daala_idtx8(const tran_low_t *input, tran_low_t *output);
void daala_fdct16(const tran_low_t *input, tran_low_t *output);
void daala_idct16(const tran_low_t *input, tran_low_t *output);
void daala_fdst16(const tran_low_t *input, tran_low_t *output);
void daala_idst16(const tran_low_t *input, tran_low_t *output);
void daala_idtx16(const tran_low_t *input, tran_low_t *output);
void daala_fdct32(const tran_low_t *input, tran_low_t *output);
void daala_idct32(const tran_low_t *input, tran_low_t *output);
void daala_fdst32(const tran_low_t *input, tran_low_t *output);
void daala_idst32(const tran_low_t *input, tran_low_t *output);
void daala_idtx32(const tran_low_t *input, tran_low_t *output);

void od_bin_fdct4(od_coeff y[4], const od_coeff *x, int xstride);
void od_bin_idct4(od_coeff *x, int xstride, const od_coeff y[4]);
void od_bin_fdst4(od_coeff y[4], const od_coeff *x, int xstride);
void od_bin_idst4(od_coeff *x, int xstride, const od_coeff y[4]);
void od_bin_fidtx4(od_coeff y[4], const od_coeff *x, int xstride);
void od_bin_iidtx4(od_coeff *x, int xstride, const od_coeff y[4]);
void od_bin_fdct8(od_coeff y[8], const od_coeff *x, int xstride);
void od_bin_idct8(od_coeff *x, int xstride, const od_coeff y[8]);
void od_bin_fdst8(od_coeff y[8], const od_coeff *x, int xstride);
void od_bin_idst8(od_coeff *x, int xstride, const od_coeff y[8]);
void od_bin_fidtx8(od_coeff y[8], const od_coeff *x, int xstride);
void od_bin_iidtx8(od_coeff *x, int xstride, const od_coeff y[8]);
void od_bin_fdct16(od_coeff y[16], const od_coeff *x, int xstride);
void od_bin_idct16(od_coeff *x, int xstride, const od_coeff y[16]);
void od_bin_fdst16(od_coeff y[16], const od_coeff *x, int xstride);
void od_bin_idst16(od_coeff *x, int xstride, const od_coeff y[16]);
void od_bin_fidtx16(od_coeff y[16], const od_coeff *x, int xstride);
void od_bin_iidtx16(od_coeff *x, int xstride, const od_coeff y[16]);
void od_bin_fdct32(od_coeff y[32], const od_coeff *x, int xstride);
void od_bin_idct32(od_coeff *x, int xstride, const od_coeff y[32]);
void od_bin_fdst32(od_coeff y[32], const od_coeff *x, int xstride);
void od_bin_idst32(od_coeff *x, int xstride, const od_coeff y[32]);
void od_bin_fidtx32(od_coeff y[32], const od_coeff *x, int xstride);
void od_bin_iidtx32(od_coeff *x, int xstride, const od_coeff y[32]);
#endif
