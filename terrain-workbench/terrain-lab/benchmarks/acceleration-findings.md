# Native solver acceleration and uniform larger cells — 2026-10-18

The user's 512×512, 1 m map used RK2, second-order Rusanov, CFL=0.4,
low-Froude mode 1, 100 m³/s daily baseflow, a 50 s warm-up, and an additional
1500 m³/s boundary inflow until t=1200 s. Requested flood duration: 1500 s.
The river inlet spans only 2 m of native boundary width.

Initial reference profiling measured approximately 0.63 wall seconds per step
while advancing only about 0.012 simulated seconds per step during warm-up.
The interface did not update until completing a 60 s warm-up check window
(50 s for this scenario), so its status remained queued throughout that work.

## Changes

- Optional cached Numba kernels fuse reconstruction, face fluxes, draining,
  transport, source/sink updates, Manning drag, and CFL wave-bound evaluation.
  No fast-math, larger CFL, lower spatial order, or different RK method is used.
- Explicit `grid_factor=1/2/4` selects uniform larger cells. Initial depth and bed
  are area means, preserving initial water volume before applying obstacles.
  Sources retain total discharge; drain capacities sum; buildings use unions;
  walls snap to nearby coarse faces. This changes spatial resolution and may
  erase narrow channel details. It is not numerically equivalent to a 1 m run.
- Warm-up and flood advancement report status roughly once per wall second,
  after accepted steps. The page displays a running remaining-time estimate.

## Short comparison

Python 3.12.9, NumPy 2.4.4, Numba 0.65.1. Three alternating-order repetitions
of 0.2 simulated seconds, excluding scene construction and warmed compilation.
The old live simulation was running during most timing samples; it was cancelled
near the end. Consequently these are local diagnostic timings, not isolated
hardware benchmarks or end-to-end runtime predictions.

| Flood variant | Cells | Steps | Median wall seconds |
|---|---:|---:|---:|
| NumPy, 1 m | 262144 | 44 | 48.385 |
| Compiled, 1 m | 262144 | 44 | 9.021 |
| Compiled, 2 m | 65536 | 20 | 0.951 |
| Compiled, 4 m | 16384 | 10 | 0.119 |

Same-resolution compiled and NumPy depths, momenta, and exported face fluxes
passed the numerical comparison. Different grid sizes represent different
spatial approximations; their mass budgets were checked, not state equality.
Raw samples and complete scenario settings are in `acceleration-results.json`.
Reproduce with `benchmark_acceleration.py --scenario SCENARIO.json --output RESULT.json`.

## Validation and deployed run

110 Python tests passed, including 48 compiled/reference combinations across
boundaries, flux methods, reconstruction order, and low-Froude modes. Three
frontend test files passed. The coarsening tests also passed with the NumPy
backend forced. Tests cover initial volume, physical extent, baseflow discharge,
external source mapping, obstacle/drain edits, and rejected grid factors.

The old run was cancelled and archived before restarting the service. A new
4 m / 128×128 run uses the original physical forcing and time settings:
`maps-3f7fca3464e4`. At its 225 s exported frame it had completed 12310 steps,
had no negative raw depths or numerical water adjustment, had maximum stage
Courant 0.2433, and had mass residual -1.92e-9 m³. Running throughput was about
1.21 simulated seconds per wall second. This is a partial-run observation;
completion and later-state stability are not claimed here.
