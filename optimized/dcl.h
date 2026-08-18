#ifndef DCL_H
#define DCL_H

#include <stdio.h>
#include <stdlib.h>

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

// ---------------------------------------------------------------------------
// Lab 4 - Canny-style edge detection pipeline
//
// 5-stage computation on a fixed-point grayscale image:
//   1. gaussian_blur         : 3x3 Gaussian smoothing (noise reduction)
//   2. sobel_gradients       : 3x3 Sobel horizontal/vertical gradients
//   3. magnitude_direction   : |Gx|+|Gy| magnitude + quantized direction
//   4. non_max_suppression   : edge thinning along gradient direction
//   5. hysteresis_threshold  : double threshold + 8-neighbor edge linking
//
// All stencil stages use zero padding at the image borders.
// ---------------------------------------------------------------------------

#define IMG_H 128
#define IMG_W 128
#define IMG_SIZE (IMG_H * IMG_W)

// Fixed-point types. Widths are chosen so that every arithmetic operation in
// the pipeline is exact (no rounding), which makes the baseline, optimized
// design, and testbench reference bit-identical by construction.
//
//   pix_t  : pixels, normalized to [0, 1), 18 fractional bits
//   grad_t : Sobel gradients, |g| <= 8,   19 fractional bits
//   mag_t  : gradient magnitude, <= 16,   20 fractional bits
//   acc_t  : stencil accumulator,         20 fractional bits
typedef ap_fixed<20, 2> pix_t;
typedef ap_fixed<24, 5> grad_t;
typedef ap_fixed<26, 6> mag_t;
typedef ap_fixed<28, 8> acc_t;
typedef ap_uint<2>      dir_t;

// Double-threshold constants for hysteresis (exact in mag_t).
#define MAG_TH ((mag_t)1.0)
#define MAG_TL ((mag_t)0.25)
#define PIX_ONE ((pix_t)1.0)

void top_kernel(pix_t in[IMG_SIZE], pix_t out[IMG_SIZE]);

#endif // DCL_H
