# Refinement parameter experiment — 2026-10-17

Production solver and defaults were not changed. Run with the user-provided
D2L virtual environment: Python 3.12.9, NumPy 2.4.4.

## Method

`benchmark_refinement.py` tests a 48×48 base grid, dx=1 m, CFL=0.4,
second-order Rusanov + RK2, zero Manning friction. Cases: sloping lake with
a dry shoreline, dam break (2 m / 0.001 m), periodic smooth wave.
Each run advances 1 simulated second; timings exclude construction and export.
Three repetitions with alternating factor order; reported times are medians.
Factors 0.5/0.75/1.0 multiply CFL*minimum_width/wave_bound; production uses 0.5.
The experimental subclass temporarily scales CFL only during timestep selection.
Buffers are 0/1/2/4 eight-neighbour dilation rings around synthetic river/lake classes.
Same-mesh factor=0.25 runs provide temporal comparison, not an exact solution.
Buffer=4/factor=0.25 projected depths provide a spatial comparison, not ground truth.

## Dam break: buffer fixed at 1

| Factor | Steps | Median seconds | Max face Courant | Depth MAE vs factor 0.25 (m) |
|---|---:|---:|---:|---:|
| 0.5 | 135 | 1.1832 | 0.400031 | 6.75e-6 |
| 0.75 | 90 | 0.8163 | 0.600108 | 1.82e-5 |
| 1.0 | 67 | 0.6091 | 0.800263 | 3.63e-5 |

## Dam break: factor fixed at 0.5

| Buffer | Leaf cells | Steps | Median seconds | Projected depth MAE vs buffer 4 / factor 0.25 (m) |
|---|---:|---:|---:|---:|
| 0 | 3264 | 135 | 1.0282 | 0.002035 |
| 1 | 3636 | 135 | 1.1832 | 0.001828 |
| 2 | 4008 | 135 | 1.2644 | 0.001378 |
| 4 | 4752 | 135 | 1.4355 | 0.00000759 |

Buffer 1→0 saved 13.1%; factor 0.5→0.75 saved 31.0%; 0.5→1.0 saved 48.5%.
These are small synthetic-case timings, not production map performance predictions.

## Stability evidence

All 36 parameter combinations completed. No negative raw depth was observed;
maximum relative mass residual was 1.98e-16. Lake depth was unchanged.
Short runs do not establish long-duration stability or accuracy on a real map.
Even factor 0.5 slightly exceeded the nominal face Courant cap in the dam case.
This indicates the current cell-based wave estimate is not a strict stage-face bound.

Existing `test_refinement` suite was also run with each factor, using the same
temporary CFL scaling around the original timestep method:

- 0.5: 11/11 passed.
- 0.75: 10/11 passed; dam-break test failed its Courant<=0.4 assertion (0.429710).
- 1.0: 10/11 passed; same assertion failed (0.575085).

The failed test stops at its first assertion failure; later method iterations in
that test are not claimed as verified. The failures are CFL contract violations,
not observed divergence.

## Interpretation

The factor 0.5 has a geometric rationale under the existing diagnostic: for a
stationary square of width w and wave speed c, the face sum is 4*c/w, while the
timestep wave bound is 2*c. Factor 0.5 gives dt*4*c/w=CFL; factor 1 gives 2*CFL.
Thus it should not simply be removed as redundant.

Keep the production factor unchanged pending a face-based timestep bound and
stage checks. Buffer 0 is a useful lower-cost option to evaluate on the actual
map, but moves the coarse/fine interface and changes the spatial solution.
The longer-term performance opportunity is reducing unnecessary global timestep
restriction and per-step leaf-mesh work while preserving conservation.

Raw data: `refinement-results.json`. Reproduce from repository root:

```powershell
& 'C:\Users\李\Desktop\D2L\.venv\Scripts\python.exe' terrain-workbench/terrain-lab/benchmarks/benchmark_refinement.py --output terrain-workbench/terrain-lab/benchmarks/refinement-results.json
```
