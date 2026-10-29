"""Public entry including the independent-parent null diagnostic recipe."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from signal_routes.common import DEFAULT,write,digest
from signal_routes.run import register,offline,full
from signal_routes.data import prepare
from signal_routes_independent_null import main as independent_null
from signal_routes_result_io import install_result_io_adapter


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,default=DEFAULT)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(); study=args.study.resolve()
    if not 1<=args.workers<=4: raise ValueError('Use 1–4 workers')
    register(study)
    write(study/'launcher.json',dict(launcher_sha256=digest(__file__),
        null_recipe_sha256=digest(Path(__file__).with_name('signal_routes_independent_null.py'))))
    prepare(study/'data'); offline(study); independent_null(study)
    install_result_io_adapter(study)
    full(study,args.workers)


if __name__=='__main__': main()
