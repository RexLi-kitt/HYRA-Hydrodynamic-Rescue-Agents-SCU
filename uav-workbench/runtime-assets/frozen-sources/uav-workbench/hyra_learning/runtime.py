"""Select already installed local packages without modifying global Python."""
from pathlib import Path
import sys


def enable_existing_packages():
    # Python 3.14 cannot load wheels from the scientific Python 3.12 runtime.
    # This path is an existing 3.14 installation verified on this workstation.
    package_dir = Path('D:/Lib/site-packages')
    if sys.version_info[:2] == (3, 14) and package_dir.is_dir() and str(package_dir) not in sys.path:
        sys.path.append(str(package_dir))
