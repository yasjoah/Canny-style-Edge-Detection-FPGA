# Canny-Style Edge Detection Accelerator

## Computation

A 5-stage edge-detection pipeline on a 128×128 fixed-point grayscale image
(a classic image-processing workload with real, chained data dependencies —
each kernel consumes the previous kernel's output):

| # | Kernel | Description |
|---|--------|-------------|
| 1 | `gaussian_blur` | 3×3 Gaussian smoothing `[1 2 1; 2 4 2; 1 2 1]/16`, zero padding |
| 2 | `sobel_gradients` | 3×3 Sobel stencils producing horizontal (Gx) and vertical (Gy) gradients |
| 3 | `magnitude_direction` | L1 magnitude `\|Gx\|+\|Gy\|` and gradient direction quantized to 4 sectors using only comparisons |
| 4 | `non_max_suppression` | Edge thinning: keep a pixel only if it is a local maximum along its gradient direction |
| 5 | `hysteresis_threshold` | Double threshold (strong/weak) + 8-neighbor edge linking, producing a binary edge map |

### Data types (identical in baseline and optimized)

All data is `ap_fixed` (see `dcl.h`):

- `pix_t  = ap_fixed<20, 2>` – pixels in [0, 1)
- `grad_t = ap_fixed<24, 5>` – Sobel gradients
- `mag_t  = ap_fixed<26, 6>` – gradient magnitudes
- `acc_t  = ap_fixed<28, 8>` – stencil accumulators
- `dir_t  = ap_uint<2>`      – quantized gradient direction

Widths are chosen so every arithmetic operation is exact (no rounding), so
the baseline, the optimized design, and the testbench reference model are
bit-identical by construction.

## Directory layout

```
lab4/
  baseline/    dcl.h  top.cpp  host.cpp  script.tcl  makefile
  optimized/   dcl.h  top.cpp  host.cpp  script.tcl  makefile
  DESIGN.md    (this file)
  REPORT_TEMPLATE.md
```

`dcl.h` and `host.cpp` are byte-identical in both folders; each folder is a
self-contained Vitis HLS project.

## Baseline (`baseline/top.cpp`)

- The five kernels run **sequentially**, communicating through full-frame
  on-chip arrays.
- Plain nested loops, no manual buffering, no HLS optimization pragmas.
- The only pragmas are the `interface` pragmas on `top_kernel` (`m_axi` for
  the two data ports + `s_axilite` for control). These define the I/O
  protocol and are **identical** in both designs, so they do not skew the
  comparison in either direction.

## Optimized (`optimized/top.cpp`)

Same five kernels, bit-exact math, restructured for performance:

- `#pragma HLS dataflow`: all five kernels (plus input/output movers) run
  **concurrently**, connected by `hls::stream` FIFOs.
- Each 3×3 stencil kernel uses a 2-row **line buffer** + 3×3 register window
  and is **pipelined at II=1** (one pixel per cycle).
- Input/output are moved with sequential `m_axi` accesses, which infer AXI
  bursts.

Expected effect: the baseline pays (number of stages) × (per-stage latency),
with unpipelined stencil loops and per-pixel DRAM accesses in stage 1. The
optimized design processes the whole frame in roughly
`(H+1)×(W+1) ≈ 16.6k` cycles end-to-end. Expected speedup is on the order
of 100×; report the exact number from co-simulation.

## Testbench (`host.cpp`, shared)

- Generates a deterministic synthetic image: 32×32-block checkerboard (two
  gray levels → strong edges) + small LCG noise.
- Computes a golden edge map with a software reference model that uses the
  same fixed-point types and operations.
- Calls `top_kernel` and compares all 16384 outputs **exactly**; prints
  `TEST PASSED` / `TEST FAILED`.

## How to run (on the class server)

For **each** of `lab4/baseline` and `lab4/optimized`:

```bash
# 1. C simulation (must print TEST PASSED)
make
./result

# 2. Full HLS flow: csynth + cosim + implementation
vitis-run --tcl script.tcl
```

## Where to get the report numbers

| Quantity | Baseline | Optimized |
|----------|----------|-----------|
| Cycle count (co-sim) | `baseline/project_baseline/hls/sim/report/top_kernel_cosim.rpt` | `optimized/project_optimized/hls/sim/report/top_kernel_cosim.rpt` |
| Clock period + resources (post-impl) | `baseline/project_baseline/hls/impl/report/verilog/top_kernel_export.rpt` | `optimized/project_optimized/hls/impl/report/verilog/top_kernel_export.rpt` |

Then:

```
latency  = cycle_count × achieved_clock_period
speedup  = baseline_latency / optimized_latency
```

