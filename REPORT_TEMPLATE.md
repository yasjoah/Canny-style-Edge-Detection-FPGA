# Lab 4 Performance Report

**Name:**
**Design:** 5-stage Canny-style edge detection pipeline (128×128, `ap_fixed`)

## 1. Design summary

- Five kernels: Gaussian blur → Sobel gradients → magnitude/direction →
  non-maximum suppression → hysteresis threshold.
- Baseline: sequential kernels with full-frame buffers, no optimization
  pragmas.
- Optimized: DATAFLOW + streaming line buffers, all stages pipelined at
  II=1, burst AXI input/output. Bit-exact with the baseline (verified by
  the shared testbench).

## 2. C/RTL co-simulation cycle counts

### Baseline

*(screenshot of `project_baseline/hls/sim/report/top_kernel_cosim.rpt`)*

- Cycle count: `_________`

### Optimized

*(screenshot of `project_optimized/hls/sim/report/top_kernel_cosim.rpt`)*

- Cycle count: `_________`

## 3. Post-implementation results

### Baseline

*(screenshot of `project_baseline/hls/impl/report/verilog/top_kernel_export.rpt`)*

- Achieved clock period (CP): `_________` ns
- Resources: LUT `_____`  FF `_____`  BRAM `_____`  DSP `_____`

### Optimized

*(screenshot of `project_optimized/hls/impl/report/verilog/top_kernel_export.rpt`)*

- Achieved clock period (CP): `_________` ns
- Resources: LUT `_____`  FF `_____`  BRAM `_____`  DSP `_____`

## 4. Speedup

| | Cycles | CP (ns) | Latency (µs) |
|---|---|---|---|
| Baseline  | | | |
| Optimized | | | |

```
speedup = baseline_latency / optimized_latency = _________
```

## 5. Correctness

- C simulation: TEST PASSED (baseline and optimized, shared testbench with
  exact fixed-point golden model).
- C/RTL co-simulation: PASSED for both designs.
