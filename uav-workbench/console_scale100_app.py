"""Compatibility entry point. Current HTTP implementation: hyra_workbench.server."""
from hyra_workbench.paths import ROOT
from hyra_workbench.server import Handler, JOBS, main

RUNNING = JOBS.lock

if __name__ == '__main__':
    main()
