"""Resolve shipped runtime inputs without rewriting frozen evidence bytes."""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUNDLE = REPO / 'uav-workbench/runtime-assets'


def runtime_path(value):
    text = str(value).replace('\\', '/')
    marker = 'HYRA-Hydrodynamic-Rescue-Agents-SCU/'
    if marker in text:
        text = text.split(marker, 1)[1]
    else:
        try:
            text = Path(text).relative_to(REPO).as_posix()
        except ValueError:
            pass
    relative = Path(text)
    if not relative.is_absolute() and ':' not in text:
        shipped = BUNDLE / relative
        if shipped.exists():
            return shipped
        return REPO / relative
    return Path(value)


def canonical_sources(records):
    return {name.replace('\\', '/'): digest for name, digest in records.items()}
