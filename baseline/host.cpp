#include "dcl.h"

// ===========================================================================
// Testbench (shared by the baseline and optimized projects).
//
// Generates a deterministic synthetic test image, computes a golden edge map
// with a software reference model (same fixed-point types and operations as
// the kernels), calls top_kernel, and compares the two results exactly.
// ===========================================================================

// Static storage to keep large arrays off the stack.
static pix_t  tb_in[IMG_SIZE];
static pix_t  tb_out_hw[IMG_SIZE];
static pix_t  tb_out_ref[IMG_SIZE];

static pix_t  ref_blur[IMG_SIZE];
static grad_t ref_gx[IMG_SIZE];
static grad_t ref_gy[IMG_SIZE];
static mag_t  ref_mag[IMG_SIZE];
static dir_t  ref_dir[IMG_SIZE];
static mag_t  ref_nms[IMG_SIZE];

// ---------------------------------------------------------------------------
// Reference model (software golden): identical math to the kernels.
// ---------------------------------------------------------------------------
static void ref_gaussian_blur(const pix_t in[IMG_SIZE], pix_t blur[IMG_SIZE])
{
    const int G[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };
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
            blur[r * IMG_W + c] = (pix_t)(acc >> 4);
        }
    }
}

static void ref_sobel_gradients(const pix_t blur[IMG_SIZE],
                                grad_t gx[IMG_SIZE], grad_t gy[IMG_SIZE])
{
    const int SX[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    const int SY[3][3] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };
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

static void ref_magnitude_direction(const grad_t gx[IMG_SIZE],
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

static void ref_non_max_suppression(const mag_t mag[IMG_SIZE],
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

static void ref_hysteresis_threshold(const mag_t nms[IMG_SIZE],
                                     pix_t out[IMG_SIZE])
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
// Deterministic synthetic test image: a checkerboard of 32x32 blocks (two
// gray levels, giving strong edges at block boundaries) plus low-amplitude
// pseudo-random noise from an LCG.
// ---------------------------------------------------------------------------
static void generate_input(pix_t in[IMG_SIZE])
{
    unsigned int lcg = 12345u;
    for (int r = 0; r < IMG_H; r++) {
        for (int c = 0; c < IMG_W; c++) {
            lcg = lcg * 1664525u + 1013904223u;
            int noise = (int)((lcg >> 24) & 0x0F);
            int base = (((r >> 5) ^ (c >> 5)) & 1) ? 200 : 60;
            int val = base + noise; // 0..255
            in[r * IMG_W + c] = (pix_t)((double)val / 256.0);
        }
    }
}

int main()
{
    generate_input(tb_in);

    // Golden reference: the same 5-stage pipeline in software.
    ref_gaussian_blur(tb_in, ref_blur);
    ref_sobel_gradients(ref_blur, ref_gx, ref_gy);
    ref_magnitude_direction(ref_gx, ref_gy, ref_mag, ref_dir);
    ref_non_max_suppression(ref_mag, ref_dir, ref_nms);
    ref_hysteresis_threshold(ref_nms, tb_out_ref);

    // Hardware kernel under test.
    top_kernel(tb_in, tb_out_hw);

    // Exact comparison.
    int mismatches = 0;
    int edge_pixels = 0;
    for (int i = 0; i < IMG_SIZE; i++) {
        if (tb_out_ref[i] == PIX_ONE) {
            edge_pixels++;
        }
        if (tb_out_hw[i] != tb_out_ref[i]) {
            if (mismatches < 10) {
                printf("Mismatch at (%d, %d): HLS = %f, Ref = %f\n",
                       i / IMG_W, i % IMG_W,
                       tb_out_hw[i].to_double(), tb_out_ref[i].to_double());
            }
            mismatches++;
        }
    }

    printf("Edge pixels in golden output: %d / %d\n", edge_pixels, IMG_SIZE);

    if (mismatches != 0) {
        printf("TEST FAILED: %d mismatches\n", mismatches);
        return 1;
    }
    printf("TEST PASSED\n");
    return 0;
}
