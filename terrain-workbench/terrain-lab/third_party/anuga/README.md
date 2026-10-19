# ANUGA source attribution

Upstream: https://github.com/anuga-community/anuga_core

Pinned commit: `e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b`.

`gpu_device_helpers.h` is an unmodified copy of
`anuga/shallow_water/gpu/gpu_device_helpers.h` from that commit.
Git blob: `9f2c3a7619899955f42ce62848839c590fdc826c`.
It generates independent scalar weir test vectors; it is not a runtime dependency.

`../../hydraulic_structures.py:weir_flux` is a Python translation of
`gpu_adjust_edgeflux_with_weir`, with fixed upstream default blend parameters.
The Cartesian coupling, geometry, schedules, limiter and UI are HYRA code.

Copyright 2004 - 2015 Australian National University and Geoscience Australia.
The upstream notice and disclaimer are retained in `LICENSE.txt`.
The full Apache License 2.0 is in `APACHE-2.0.txt`.
