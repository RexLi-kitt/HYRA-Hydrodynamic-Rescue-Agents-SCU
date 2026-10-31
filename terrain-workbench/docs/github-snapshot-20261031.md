# Carlisle research evidence snapshot — 2026-10-31

This snapshot extends the merged PR10 with numerical-failure replay and isolated reference-backend patches, complete Carlisle event audit, and missed-inundation diagnosis through round95. It does not promote a reference backend into the production simulator.

The completed5m LISFLOOD-FP8 SGC/local-inertia reference run covered192580 of217464 eligible observed-wet cells (88.56%). There are24884 retained misses. This is positive coverage, not overall accuracy: reliable observed-dry coverage, road usability and rescue safety validation remain incomplete. The observed-field comparisons are conditional on the archived event/time/terrain assumptions documented with round91.

The southern component contains1457 missed cells. Round95 static reachability is an optimistic non-conservative diagnostic, not another flood prediction; it does not improve the88.56% score. The historical geometry and mechanism gaps remain unresolved.

## Contents and reproduction limits

Included file hashes and an explicit local-only exclusion list are in github-snapshot-20261031.json. Existing per-round manifests describe full local archives; some referenced files are intentionally not shipped here. Third-party PDFs/images/source replicas, original DEM/boundary archives, large raw simulation archives and binary array files remain on the research machine. Their source links and hashes remain available for provenance. Existing legacy absolute paths in archived scripts require path configuration and appropriate input acquisition before reproduction.

Source baselines and upstream provenance for isolated patches are recorded in their per-round reports. Patches are research candidates, not applied changes to the production solver. No third-party license is changed or superseded by this snapshot.

## Validation

Python scripts parsed successfully. The recorded first-failure algebraic replay is rerun from the included trace before submission. Existing flood-platform tests are run separately, and their results are reported in the PR. Full hydraulic experiments are not rerun for upload. Original reference-run mass/finite/nonnegative audits and the round95 independent-method comparison are archived evidence, not new checks executed by GitHub.
