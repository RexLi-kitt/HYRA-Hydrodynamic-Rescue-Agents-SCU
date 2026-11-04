# Round77 RK2 boundary ledger audit (2026-10-30)

Apply rk-boundary-ledger.patch after round76 and rebuild the isolated CPU64 DG2 adapter. Each stage contributes half its gross boundary inflow/outflow; accepted point-source volume is counted once. Seven small closed/open/mixed controls pass, all emitted physical fields and accepted steps remain byte-identical. This fixes accounting, not flood predictions. No production adoption or field-validation claim. See docs/audit-round77-20261030.md and outputs/round77. Rain, restart, negative sources, multithreading and complete event remain outside the tested scope.

# Round76 accepted-interval point-source integration (2026-10-30)

Apply `accepted-source-integral.patch` after round75's isolated observer/reconstruction patches and fully rebuild CPU64. Nonnegative QVAR is integrated exactly over each proposed interval, splitting original time knots and preserving the existing constant endpoint extension. Retry restores original source-cell H and recomputes the interval; commit once, using volume/dt in the mass ledger. QFIX, source locations, q*dx semantics and source momentum assumptions remain unchanged. Source audit records are interval totals, not instantaneous observations.

Actual-function analytic/partition/forced-retry controls pass; three stationary-water and constant-source full fields remain identical. Original60s Carlisle input admits111.0936m3 at both tested max timesteps, with finite nonnegative depth and closed ledger. This removes the previous left-endpoint source bias, NOT all hydrodynamic time error or event-input uncertainty. No whole-event observational validation, nonzero-open-boundary certification, rain/restart/negative-source support or production adoption is claimed. See outputs/round76 and docs/audit-round76-20261030.md. Multi-aspect laboratory scores are separately scoped in qualitative-agreement-round76-20261030.md and do not transfer to this LF8 candidate.

# Round75 diagnostic observer and isolated reconstruction candidate (2026-10-30)

Apply `negative-depth-observer.patch` after round74; then optionally `nonnegative-reconstruction-candidate.patch`. Rebuild a separate CPU64 executable. Both require OMP_NUM_THREADS=1. Observer exit42 intentionally stops before a material negative RK update; it is not a completed simulation. No negative depth is clipped. The candidate rescales six DG slope coefficients, preserving all cell averages, before each RK flux evaluation. It is research-only, NOT a production default or a proof of positivity for the unchanged CFL/hydrostatic scheme.

First negative: t=25.645547s at Eden source, RK2; mean .007445m with reconstructed west1.312751/east-1.297861m; shared mass fluxes independently reproduce H=-.029093103m. Candidate fixed60s and max-dt-halved60s both have no negative traces/final cells and close volume ledgers. Three stationary-water controls and a flat multi-source control pass. See outputs/round75 and docs/audit-round75-20261030.md. Larger accepted dt exposes existing left-endpoint source integration bias; this remains an explicit next task. Complete field validation and broader dynamic/open-boundary regressions remain unperformed. Original observation rules and production defaults are unchanged.

The build/run scripts preserve the project absolute provenance paths and reject existing destination directories. Preserve completed runs; do not rerun them into the same folder.

# Round74 rectangular perimeter correction (2026-10-30)

Apply `perimeter-row-stride.patch` after the preceding patches, then fully rebuild an isolated CPU64 reference. It changes six y-coefficient indices in zero_perimeter_slopes from j*ysz+i to j*xsz+i. `verify_perimeter_stride.py` compiles the actual old/new function against bounds-checked buffers and independent boundary-value expectations for wide, tall, square and Carlisle-sized grids. The old tall kernel accesses outside the array; the old wide kernel changes interior cells. All corrected checks pass, and a square multi-source full-solver control remains bitwise identical.

This does NOT fix Carlisle startup positivity. Same-data 60s runs retain four negative-depth cells (minimum -0.014360384310m). Enabling the existing limitslopes option gives an identical final depth array and is rejected as a cure. No complete field event is admitted yet. Next use a trace-only first-negative RK/face-flux diagnostic. The nested repeated south-edge loop is preserved to isolate the row-stride fix; only OMP_NUM_THREADS=1 is verified. See outputs/round74 and docs/audit-round74-20261030.md. Production defaults are unchanged.

# Round73 stable boundary-series storage (2026-10-30)

Apply `stable-boundary-series.patch` after the round65 event and round71 accepted-point patches, in an isolated copy, then fully rebuild. `LoadBCVar` keeps pointers to appended TimeSeries objects; changing BoundCs::allTimeSeries from vector to deque preserves referenced objects across later appends and removal of an unused last element. No numerical formula is changed. Baseline source/binary remain in round71. Candidate: outputs/round73/build-stable-cpu64/Release/lisflood.exe. Single-series final fields stay bitwise identical; multiple-series permutation and unused-series insertion are invariant. The small old multi-series test does NOT reliably crash; the archived Carlisle input does, and is preserved as the integration reproducer.

The candidate now completes the 60s Carlisle startup, but its 2 negative-depth cells fail the preregistered nonnegative criterion. This is NOT a validated whole-event backend. Do not launch a full field run or upgrade production defaults. Next investigate rectangular perimeter indexing and positivity separately; do not clip output or tune observations. See audit-round73-20261030.md and outputs/round73/next-candidate-protocol.json. Existing no-rain/no-restart/nonnegative-point-flow limits remain. Scalar source integration is still first order.

# Round71 positive-point-source extension (2026-10-30)

The separate `accepted-point-step.patch` is applied AFTER round65's `exact-output-events.patch`, only to an isolated CPU FP64 DG2 copy. The source and executable in outputs/round65 remain unchanged. The accepted candidate is outputs/round71/build-point-cpu64/Release/lisflood.exe; build commands and hashes are in outputs/round71. This is not the HYRA production default and does not constitute field validation.

Scope: nonnegative QFIX/QVAR point inflows, no rain, t=0 starts. HFIX/HVAR, negative/nonfinite point flows, rainfall and restart are explicitly rejected. q in a DG2 P record is unit-width discharge (m2/s); injected volume is q*dx*accepted_dt, summed over all points, including duplicates in one cell. Preserve physical total Q when changing dx or point count; do not infer spatial placement from that rule.

The former point-source path updated H using the previous timestep, then chose a new CFL/event step for integration and accounting. The candidate snapshots source-cell depths, probes the source-added state with a bounded candidate dt, shrinks to a valid post-source CFL limit, and commits exactly once. The existing q(t_n) source quadrature remains first order; this patch does not add incoming momentum or infer river velocities. Boundary-only fixtures remain bitwise identical.

Verification: probe_point_sources.py EXE NEW_OUTPUT_DIR runs six independently integrated constant-flow fixtures. verify_point_source_regression.py ROUND71_EVIDENCE_ROOT checks the saved round65 boundary fixture, a strong dry source, two linear-source time steps and explicit exclusions; it refuses an existing regression directory. Raw before/after, failed missing-slope attempt and full logs are preserved in round71. Linear-source quadrature error halves with the timestep; it has not been eliminated. Reported field-like performance is not an acceptance criterion here.

Carlisle's original 23 points have groups11/6/6 on5m. The accompanying boundary series are inputs from documented1D routing, not independent flow observations. `../carlisle_source_contract.py T07_DIR NEW_OUTPUT_DIR` preserves all274 timestamps per series and converts each to total m3/s. Absolute event start, actual channel geometry, restricted5m bed and correct50m source locations remain unresolved. No Carlisle model is generated or validated by this conversion.

Rollback: stop using the point candidate and use the preserved round65 reference only within its old no-source/no-rain scope; do not delete old source or results.

The earlier round65 scope and audit follow unchanged.

---

# Isolated LISFLOOD-FP8 reference audit

Scope: CPU DG2, FP64, t=0 starts, closed/boundary-driven cases without rainfall or point sources. This is a reference adapter, not a new HYRA default. No full Merewether physical validation has been completed with this backend.

Official source: https://doi.org/10.5281/zenodo.4073011 . Preserve the official source; apply `exact-output-events.patch` only to a separate copy of `swe/dg2.cpp`. Before SHA256 and after SHA256 are in round65/reference-event-patch.json. Build commands, MSVC/CMake configuration and binary hashes are in round65/build-event-cpu64.json. `_NUMERIC_MODE=1` means double; 0 means float. Runtime banner was checked.

The patch caps the already CFL-limited time step at mass/sample, map and final events. It logs actual accepted times in accepted_timesteps.txt in the run directory, and starts outputs at one interval. It does not change fluxes, slopes or friction formulas. Do not use it for rainfall/point-source/restart cases before extending validation. It is not a general upstream fix.

The 400-cell flat dry inlet fixture, raw failed output, precision12 repeat and logs live in the user's outputs/round65. Run verify_event_smoke.py with before-fixture, patched-fixture and output-JSON arguments. Use reference-inlet-smoke and reference-inlet-event-precision12. Require exact .05/.10/.15/.20 s events, total .05 m3/s inflow, .01 m3 final volume, finite/nonnegative depth, dry gauge speeds zero. Default three-decimal depth export cannot support a 1e-6 m3 map-volume assertion; use official output_precision=12. Dynamics were byte-identical between precision3 and12.

Before any full40 run: resolve the fixed DG2 preprocessing coordinate convention from original sources (not score optimization), original north extent, boundary unit discharge, all original 40 points, water-depth sampling cutoff and full time grid. The official .175m configuration and legacy .7m HYRA also differ in geometry/domain; differences cannot be attributed solely to a solver formula. GPU transport and timing are not validated.

Rollback: use the saved unmodified official source/build-cpu64 binary; leave existing HYRA defaults and archived results intact.
