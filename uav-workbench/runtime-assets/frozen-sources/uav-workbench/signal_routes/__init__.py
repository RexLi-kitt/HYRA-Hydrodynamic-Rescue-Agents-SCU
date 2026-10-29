"""Isolated, preregistered signal-belief study. No legacy model files are changed."""

import os
from hyra_learning.runtime import enable_existing_packages

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
enable_existing_packages()

METHODS = ('Braw', 'Bpost', 'R1', 'R2', 'R3', 'C1', 'C2', 'C3')
SCHEMA = 'hyra-signal-routes-v2'
