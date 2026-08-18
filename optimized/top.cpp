#include "dcl.h"

// ===========================================================================
// OPTIMIZED implementation.
//
// Functionally identical to the baseline (bit-exact), but restructured for
// performance:
//   - All five kernels run concurrently under #pragma HLS dataflow.
//   - Pixels stream between stages through hls::stream FIFOs, so no
//     full-frame intermediate buffers are needed.
//   - Each 3x3 stencil stage keeps a 2-row line buffer plus a 3x3 window of
//     registers and produces one output per cycle (pipeline II=1).
//   - Input/output are moved with burst-friendly sequential m_axi accesses.
//
// Stencil iteration pattern: each stencil stage iterates over an extended
// (IMG_H+1) x (IMG_W+1) domain. At iteration (r, c) the window holds image
// rows r-2..r and columns c-2..c, and the output pixel centered at
// (r-1, c-1) is emitted when r >= 1 && c >= 1. Zero padding falls out
// naturally:
//   - top border   : line buffers are initialized to 0
//   - bottom border: nothing is read at r == IMG_H, zeros are shifted in
//   - right border : nothing is read at c == IMG_W, zeros are shifted in
//   - left border  : the leftmost window column is muxed to 0 when c == 1
// ===========================================================================

// ---------------------------------------------------------------------------
// Input mover: DRAM -> stream (sequential reads infer an AXI burst).
// ---------------------------------------------------------------------------
static void read_input(const pix_t in[IMG_SIZE], hls::stream<pix_t> &s_out)
{
READ_LOOP:
    for (int i = 0; i < IMG_SIZE; i++) {
#pragma HLS pipeline II=1
        s_out.write(in[i]);
    }
}

// ---------------------------------------------------------------------------
// Kernel 1: 3x3 Gaussian blur (streaming, line-buffered, II=1).
// Same computation as the baseline: [1 2 1; 2 4 2; 1 2 1] / 16 with zero
// padding at the borders.
// ---------------------------------------------------------------------------
static void gaussian_blur(hls::stream<pix_t> &s_in, hls::stream<pix_t> &s_out)
{
    pix_t lb0[IMG_W];
    pix_t lb1[IMG_W];

GB_INIT:
    for (int c = 0; c < IMG_W; c++) {
#pragma HLS pipeline II=1
        lb0[c] = 0;
        lb1[c] = 0;
    }

    pix_t w00 = 0, w01 = 0, w02 = 0;
    pix_t w10 = 0, w11 = 0, w12 = 0;
    pix_t w20 = 0, w21 = 0, w22 = 0;

GB_ROW:
    for (int r = 0; r <= IMG_H; r++) {
GB_COL:
        for (int c = 0; c <= IMG_W; c++) {
#pragma HLS pipeline II=1
            pix_t p = 0, c0 = 0, c1 = 0;
            if (r < IMG_H && c < IMG_W) {
                p  = s_in.read();
                c0 = lb0[c];
                c1 = lb1[c];
                lb0[c] = c1;
                lb1[c] = p;
            }
            w00 = w01; w01 = w02; w02 = c0;
            w10 = w11; w11 = w12; w12 = c1;
            w20 = w21; w21 = w22; w22 = p;

            if (r >= 1 && c >= 1) {
                pix_t a00 = (c == 1) ? (pix_t)0 : w00;
                pix_t a10 = (c == 1) ? (pix_t)0 : w10;
                pix_t a20 = (c == 1) ? (pix_t)0 : w20;

                acc_t acc = 0;
                acc += a00 * 1;
                acc += w01 * 2;
                acc += w02 * 1;
                acc += a10 * 2;
                acc += w11 * 4;
                acc += w12 * 2;
                acc += a20 * 1;
                acc += w21 * 2;
                acc += w22 * 1;

                s_out.write((pix_t)(acc >> 4));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: 3x3 Sobel gradients (streaming, line-buffered, II=1).
// Same computation as the baseline SX/SY stencils with zero padding.
// ---------------------------------------------------------------------------
static void sobel_gradients(hls::stream<pix_t> &s_in,
                            hls::stream<grad_t> &s_gx,
                            hls::stream<grad_t> &s_gy)
{
    pix_t lb0[IMG_W];
    pix_t lb1[IMG_W];

SB_INIT:
    for (int c = 0; c < IMG_W; c++) {
#pragma HLS pipeline II=1
        lb0[c] = 0;
        lb1[c] = 0;
    }

    pix_t w00 = 0, w01 = 0, w02 = 0;
    pix_t w10 = 0, w11 = 0, w12 = 0;
    pix_t w20 = 0, w21 = 0, w22 = 0;

SB_ROW:
    for (int r = 0; r <= IMG_H; r++) {
SB_COL:
        for (int c = 0; c <= IMG_W; c++) {
#pragma HLS pipeline II=1
            pix_t p = 0, c0 = 0, c1 = 0;
            if (r < IMG_H && c < IMG_W) {
                p  = s_in.read();
                c0 = lb0[c];
                c1 = lb1[c];
                lb0[c] = c1;
                lb1[c] = p;
            }
            w00 = w01; w01 = w02; w02 = c0;
            w10 = w11; w11 = w12; w12 = c1;
            w20 = w21; w21 = w22; w22 = p;

            if (r >= 1 && c >= 1) {
                pix_t a00 = (c == 1) ? (pix_t)0 : w00;
                pix_t a10 = (c == 1) ? (pix_t)0 : w10;
                pix_t a20 = (c == 1) ? (pix_t)0 : w20;

                acc_t ax = 0;
                ax += a00 * (-1);
                ax += w02 * 1;
                ax += a10 * (-2);
                ax += w12 * 2;
                ax += a20 * (-1);
                ax += w22 * 1;

                acc_t ay = 0;
                ay += a00 * 1;
                ay += w01 * 2;
                ay += w02 * 1;
                ay += a20 * (-1);
                ay += w21 * (-2);
                ay += w22 * (-1);

                s_gx.write((grad_t)ax);
                s_gy.write((grad_t)ay);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: gradient magnitude and quantized direction (streaming, II=1).
// Pointwise stage; same computation as the baseline.
// ---------------------------------------------------------------------------
static void magnitude_direction(hls::stream<grad_t> &s_gx,
                                hls::stream<grad_t> &s_gy,
                                hls::stream<mag_t> &s_mag,
                                hls::stream<dir_t> &s_dir)
{
MD_LOOP:
    for (int i = 0; i < IMG_SIZE; i++) {
#pragma HLS pipeline II=1
        grad_t x = s_gx.read();
        grad_t y = s_gy.read();

        grad_t xa = (x < 0) ? (grad_t)(-x) : x;
        grad_t ya = (y < 0) ? (grad_t)(-y) : y;
        mag_t ax = (mag_t)xa;
        mag_t ay = (mag_t)ya;

        mag_t m = ax + ay;

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

        s_mag.write(m);
        s_dir.write(d);
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: non-maximum suppression (streaming, line-buffered, II=1).
// Keeps a 3x3 window of magnitudes plus a parallel window of directions
// (only the center direction is used). Same computation as the baseline.
// ---------------------------------------------------------------------------
static void non_max_suppression(hls::stream<mag_t> &s_mag,
                                hls::stream<dir_t> &s_dir,
                                hls::stream<mag_t> &s_out)
{
    mag_t lbm0[IMG_W];
    mag_t lbm1[IMG_W];
    dir_t lbd0[IMG_W];
    dir_t lbd1[IMG_W];

NM_INIT:
    for (int c = 0; c < IMG_W; c++) {
#pragma HLS pipeline II=1
        lbm0[c] = 0;
        lbm1[c] = 0;
        lbd0[c] = 0;
        lbd1[c] = 0;
    }

    mag_t m00 = 0, m01 = 0, m02 = 0;
    mag_t m10 = 0, m11 = 0, m12 = 0;
    mag_t m20 = 0, m21 = 0, m22 = 0;
    dir_t d00 = 0, d01 = 0, d02 = 0;
    dir_t d10 = 0, d11 = 0, d12 = 0;
    dir_t d20 = 0, d21 = 0, d22 = 0;

NM_ROW:
    for (int r = 0; r <= IMG_H; r++) {
NM_COL:
        for (int c = 0; c <= IMG_W; c++) {
#pragma HLS pipeline II=1
            mag_t p = 0, c0 = 0, c1 = 0;
            dir_t pd = 0, cd0 = 0, cd1 = 0;
            if (r < IMG_H && c < IMG_W) {
                p   = s_mag.read();
                pd  = s_dir.read();
                c0  = lbm0[c];
                c1  = lbm1[c];
                cd0 = lbd0[c];
                cd1 = lbd1[c];
                lbm0[c] = c1;
                lbm1[c] = p;
                lbd0[c] = cd1;
                lbd1[c] = pd;
            }
            m00 = m01; m01 = m02; m02 = c0;
            m10 = m11; m11 = m12; m12 = c1;
            m20 = m21; m21 = m22; m22 = p;
            d00 = d01; d01 = d02; d02 = cd0;
            d10 = d11; d11 = d12; d12 = cd1;
            d20 = d21; d21 = d22; d22 = pd;

            if (r >= 1 && c >= 1) {
                mag_t a00 = (c == 1) ? (mag_t)0 : m00;
                mag_t a10 = (c == 1) ? (mag_t)0 : m10;
                mag_t a20 = (c == 1) ? (mag_t)0 : m20;

                mag_t m = m11;
                dir_t d = d11;

                mag_t n1, n2;
                if (d == 0) {
                    n1 = a10; n2 = m12;
                } else if (d == 1) {
                    n1 = a00; n2 = m22;
                } else if (d == 2) {
                    n1 = m01; n2 = m21;
                } else {
                    n1 = m02; n2 = a20;
                }

                mag_t o = (m >= n1 && m >= n2) ? m : (mag_t)0;
                s_out.write(o);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 5: double threshold + hysteresis edge linking (streaming, II=1).
// 3x3 window over the NMS output; same computation as the baseline.
// ---------------------------------------------------------------------------
static void hysteresis_threshold(hls::stream<mag_t> &s_in,
                                 hls::stream<pix_t> &s_out)
{
    mag_t lb0[IMG_W];
    mag_t lb1[IMG_W];

HT_INIT:
    for (int c = 0; c < IMG_W; c++) {
#pragma HLS pipeline II=1
        lb0[c] = 0;
        lb1[c] = 0;
    }

    mag_t w00 = 0, w01 = 0, w02 = 0;
    mag_t w10 = 0, w11 = 0, w12 = 0;
    mag_t w20 = 0, w21 = 0, w22 = 0;

HT_ROW:
    for (int r = 0; r <= IMG_H; r++) {
HT_COL:
        for (int c = 0; c <= IMG_W; c++) {
#pragma HLS pipeline II=1
            mag_t p = 0, c0 = 0, c1 = 0;
            if (r < IMG_H && c < IMG_W) {
                p  = s_in.read();
                c0 = lb0[c];
                c1 = lb1[c];
                lb0[c] = c1;
                lb1[c] = p;
            }
            w00 = w01; w01 = w02; w02 = c0;
            w10 = w11; w11 = w12; w12 = c1;
            w20 = w21; w21 = w22; w22 = p;

            if (r >= 1 && c >= 1) {
                mag_t a00 = (c == 1) ? (mag_t)0 : w00;
                mag_t a10 = (c == 1) ? (mag_t)0 : w10;
                mag_t a20 = (c == 1) ? (mag_t)0 : w20;

                mag_t v = w11;
                bool strong = (v > MAG_TH);
                bool weak   = (v > MAG_TL);

                bool any_strong = false;
                if (a00 > MAG_TH) any_strong = true;
                if (w01 > MAG_TH) any_strong = true;
                if (w02 > MAG_TH) any_strong = true;
                if (a10 > MAG_TH) any_strong = true;
                if (w12 > MAG_TH) any_strong = true;
                if (a20 > MAG_TH) any_strong = true;
                if (w21 > MAG_TH) any_strong = true;
                if (w22 > MAG_TH) any_strong = true;

                pix_t o = 0;
                if (strong) {
                    o = PIX_ONE;
                } else if (weak && any_strong) {
                    o = PIX_ONE;
                }
                s_out.write(o);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Output mover: stream -> DRAM (sequential writes infer an AXI burst).
// ---------------------------------------------------------------------------
static void write_output(hls::stream<pix_t> &s_in, pix_t out[IMG_SIZE])
{
WRITE_LOOP:
    for (int i = 0; i < IMG_SIZE; i++) {
#pragma HLS pipeline II=1
        out[i] = s_in.read();
    }
}

// ---------------------------------------------------------------------------
// Top function: all five kernels run concurrently under DATAFLOW, connected
// by FIFOs. Total latency is roughly one frame time (~(IMG_H+1)*(IMG_W+1)
// cycles) instead of the sum of five sequential stage latencies.
// ---------------------------------------------------------------------------
void top_kernel(pix_t in[IMG_SIZE], pix_t out[IMG_SIZE])
{
#pragma HLS interface m_axi port=in offset=slave bundle=gmem_in depth=16384
#pragma HLS interface m_axi port=out offset=slave bundle=gmem_out depth=16384
#pragma HLS interface s_axilite port=return
#pragma HLS dataflow

    hls::stream<pix_t>  s_raw("s_raw");
    hls::stream<pix_t>  s_blur("s_blur");
    hls::stream<grad_t> s_gx("s_gx");
    hls::stream<grad_t> s_gy("s_gy");
    hls::stream<mag_t>  s_mag("s_mag");
    hls::stream<dir_t>  s_dir("s_dir");
    hls::stream<mag_t>  s_nms("s_nms");
    hls::stream<pix_t>  s_edge("s_edge");
#pragma HLS stream variable=s_raw  depth=16
#pragma HLS stream variable=s_blur depth=16
#pragma HLS stream variable=s_gx   depth=16
#pragma HLS stream variable=s_gy   depth=16
#pragma HLS stream variable=s_mag  depth=16
#pragma HLS stream variable=s_dir  depth=16
#pragma HLS stream variable=s_nms  depth=16
#pragma HLS stream variable=s_edge depth=16

    read_input(in, s_raw);
    gaussian_blur(s_raw, s_blur);
    sobel_gradients(s_blur, s_gx, s_gy);
    magnitude_direction(s_gx, s_gy, s_mag, s_dir);
    non_max_suppression(s_mag, s_dir, s_nms);
    hysteresis_threshold(s_nms, s_edge);
    write_output(s_edge, out);
}
