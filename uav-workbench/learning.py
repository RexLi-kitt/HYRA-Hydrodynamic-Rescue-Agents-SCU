"""Entry point for the independent, registered Sur-QMIX adaptation preflight."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()

if __name__ == '__main__':
    from hyra_learning.training import main
    main()
