#include "dcl.h"

// ===========================================================================
// BASELINE implementation.
//
// Straightforward, sequential code. The five kernels execute one after
// another, each reading/writing full-frame arrays. There are NO optimization
// pragmas anywhere in this file; the only pragmas are the interface pragmas
// on the top function, which define the I/O protocol and are identical in
// the baseline and optimized designs.
// ===========================================================================

// ---------------------------------------------------------------------------
// Kernel 1: 3x3 Gaussian blur.
// Smooths the input image with the normalized kernel
//   [1 2 1; 2 4 2; 1 2 1] / 16
// using zero padding at the borders.
// ---------------------------------------------------------------------------
static void gaussian_blur(const pix_t in[IMG_SIZE], pix_t blur[IMG_SIZE])
{
    const int G[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    for (int r = 0; r < IMG_H; r++) {
        for (int c = 0; c < IMG_W; c++) {
            acc_t acc = 0;
            for (int kr = -1; kr <= 1; kr++) {
                for (int kc = -1; kc <= 1; kc++) {
                    int rr = r + kr;
                    int cc = c + kc;
                    pix_t v = 0;
                    if (rr >= 0 && rr < IMG_H && cc >= 0 && cc < IMG_W) {
                        v = in[rr * IMG_W + cc];
                    }
                    acc += v * G[kr + 1][kc + 1];
                }
            }
            // Divide by 16 (kernel weight sum).
            blur[r * IMG_W + c] = (pix_t)(acc >> 4);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: 3x3 Sobel gradients.
// Computes the horizontal gradient Gx and vertical gradient Gy of the
// blurred image, with zero padding at the borders.
// ---------------------------------------------------------------------------
static void sobel_gradients(const pix_t blur[IMG_SIZE],
                            grad_t gx[IMG_SIZE], grad_t gy[IMG_SIZE])
{
    const int SX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    const int SY[3][3] = {
        { 1,  2,  1},
        { 0,  0,  0},
        {-1, -2, -1}
    };

    for (int r = 0; r < IMG_H; r++) {
        for (int c = 0; c < IMG_W; c++) {
            acc_t ax = 0;
            acc_t ay = 0;
            for (int kr = -1; kr <= 1; kr++) {
                for (int kc = -1; kc <= 1; kc++) {
                    int rr = r + kr;
                    int cc = c + kc;
                    pix_t v = 0;
                    if (rr >= 0 && rr < IMG_H && cc >= 0 && cc < IMG_W) {
                        v = blur[rr * IMG_W + cc];
                    }
                    ax += v * SX[kr + 1][kc + 1];
                    ay += v * SY[kr + 1][kc + 1];
                }
            }
            gx[r * IMG_W + c] = (grad_t)ax;
            gy[r * IMG_W + c] = (grad_t)ay;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: gradient magnitude and quantized direction.
// Magnitude uses the L1 approximation |Gx| + |Gy|. Direction is quantized
// into 4 sectors using only comparisons (no atan needed):
//   0 : mostly horizontal gradient (|Gx| >= 2|Gy|)  -> vertical edge
//   2 : mostly vertical gradient   (|Gy| >= 2|Gx|)  -> horizontal edge
//   1 : 45-degree diagonal (Gx and Gy same sign)
//   3 : 135-degree diagonal (Gx and Gy opposite sign)
// ---------------------------------------------------------------------------
static void magnitude_direction(const grad_t gx[IMG_SIZE],
                                const grad_t gy[IMG_SIZE],
                                mag_t mag[IMG_SIZE], dir_t dir[IMG_SIZE])
{
    for (int r = 0; r < IMG_H; r++) {
        for (int c = 0; c < IMG_W; c++) {
            int idx = r * IMG_W + c;
            grad_t x = gx[idx];
            grad_t y = gy[idx];

            grad_t xa = (x < 0) ? (grad_t)(-x) : x;
            grad_t ya = (y < 0) ? (grad_t)(-y) : y;
            mag_t ax = (mag_t)xa;
            mag_t ay = (mag_t)ya;

            mag[idx] = ax + ay;

            dir_t d;
            if (ax >= ay + ay) {
                d = 0;
            } else if (ay >= ax + ax) {
                d = 2;
            } else if ((x > 0) == (y > 0)) {
                d = 1;
            } else {
                d = 3;
            }
            dir[idx] = d;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: non-maximum suppression.
// A pixel keeps its magnitude only if it is a local maximum along its
// gradient direction; otherwise it is suppressed to 0. Out-of-image
// neighbors are treated as 0.
// ---------------------------------------------------------------------------
static void non_max_suppression(const mag_t mag[IMG_SIZE],
                                const dir_t dir[IMG_SIZE],
                                mag_t nms[IMG_SIZE])
{
    for (int r = 0; r < IMG_H; r++) {
        for (int c = 0; c < IMG_W; c++) {
            int idx = r * IMG_W + c;
            mag_t m = mag[idx];
            dir_t d = dir[idx];

            int r1, c1, r2, c2;
            if (d == 0) {
                r1 = r;     c1 = c - 1; r2 = r;     c2 = c + 1;
            } else if (d == 1) {
                r1 = r - 1; c1 = c - 1; r2 = r + 1; c2 = c + 1;
            } else if (d == 2) {
                r1 = r - 1; c1 = c;     r2 = r + 1; c2 = c;
            } else {
                r1 = r - 1; c1 = c + 1; r2 = r + 1; c2 = c - 1;
            }

            mag_t n1 = 0;
            if (r1 >= 0 && r1 < IMG_H && c1 >= 0 && c1 < IMG_W) {
                n1 = mag[r1 * IMG_W + c1];
            }
            mag_t n2 = 0;
            if (r2 >= 0 && r2 < IMG_H && c2 >= 0 && c2 < IMG_W) {
                n2 = mag[r2 * IMG_W + c2];
            }

            nms[idx] = (m >= n1 && m >= n2) ? m : (mag_t)0;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 5: double threshold + hysteresis edge linking.
// Pixels above MAG_TH are strong edges. Pixels between MAG_TL and MAG_TH
// are weak edges and are kept only if at least one of their 8 neighbors is
// a strong edge. Output is a binary edge map (1.0 or 0.0).
// ---------------------------------------------------------------------------
static void hysteresis_threshold(const mag_t nms[IMG_SIZE], pix_t out[IMG_SIZE])
{
    for (int r = 0; r < IMG_H; r++) {
        for (int c = 0; c < IMG_W; c++) {
            int idx = r * IMG_W + c;
            mag_t v = nms[idx];

            bool strong = (v > MAG_TH);
            bool weak   = (v > MAG_TL);

            bool any_strong = false;
            for (int kr = -1; kr <= 1; kr++) {
                for (int kc = -1; kc <= 1; kc++) {
                    if (kr == 0 && kc == 0) {
                        continue;
                    }
                    int rr = r + kr;
                    int cc = c + kc;
                    mag_t n = 0;
                    if (rr >= 0 && rr < IMG_H && cc >= 0 && cc < IMG_W) {
                        n = nms[rr * IMG_W + cc];
                    }
                    if (n > MAG_TH) {
                        any_strong = true;
                    }
                }
            }

            pix_t o = 0;
            if (strong) {
                o = PIX_ONE;
            } else if (weak && any_strong) {
                o = PIX_ONE;
            }
            out[idx] = o;
        }
    }
}

// ---------------------------------------------------------------------------
// Top function: runs the five kernels sequentially through full-frame
// intermediate buffers.
// ---------------------------------------------------------------------------
void top_kernel(pix_t in[IMG_SIZE], pix_t out[IMG_SIZE])
{
#pragma HLS interface m_axi port=in offset=slave bundle=gmem_in depth=16384
#pragma HLS interface m_axi port=out offset=slave bundle=gmem_out depth=16384
#pragma HLS interface s_axilite port=return

    pix_t  blur[IMG_SIZE];
    grad_t gx[IMG_SIZE];
    grad_t gy[IMG_SIZE];
    mag_t  mag[IMG_SIZE];
    dir_t  dir[IMG_SIZE];
    mag_t  nms[IMG_SIZE];

    gaussian_blur(in, blur);
    sobel_gradients(blur, gx, gy);
    magnitude_direction(gx, gy, mag, dir);
    non_max_suppression(mag, dir, nms);
    hysteresis_threshold(nms, out);
}
