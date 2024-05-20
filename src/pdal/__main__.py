import sys
import os
import pathlib

import sysconfig

import argparse

import pdal

from . import __version__

__all__ = ["main"]


def __dir__() -> list[str]:
    return __all__


def print_driver_path(args):
    if 'PDAL_DRIVER_PATH' in os.environ:
        print (os.environ['PDAL_DRIVER_PATH'])

def print_plugin_path(args):
    purelib = sysconfig.get_paths()["purelib"]

    if sys.platform == "linux" or sys.platform == "linux2":
        suffix = 'so'
        purelib = purelib + os.path.sep + "pdal"
    elif sys.platform == "darwin":
        suffix = 'dylib'
        purelib = purelib + os.path.sep + "pdal"
    elif sys.platform == "win32":
        suffix = 'dll'
        purelib = purelib + os.path.sep + "bin"

    for f in pathlib.Path(purelib).glob(f'*.{suffix}'):
        if 'pdal' in str(f.name):
            if 'numpy' in str(f.name) or 'python' in str(f.name):
                print (purelib)
                return # we are done

def print_version(args):
    info = pdal.drivers.libpdalpython.getInfo()
    pdal_version = info.version
    plugin = info.plugin
    debug = info.debug

    line = '----------------------------------------------------------------------------------------------------------------------------\n'
    version = f'PDAL version {pdal_version}\nPython bindings version {__version__}\n'
    plugin = f"Environment-set PDAL_DRIVER_PATH: {os.environ['PDAL_DRIVER_PATH']}"
    output = f'{line}{version}{plugin}\n{line}\n{debug}'
    print (output)


def main() -> None:
    header = f"PDAL Python bindings {__version__} on Python {sys.version}"

    parser = argparse.ArgumentParser(description=header)
    parser.add_argument('--pdal-driver-path',  action='store_true',
                        help='print PDAL_DRIVER_PATH including Python plugin locations')
    parser.add_argument('--pdal-plugin-path',  action='store_true',
                        help='print location of PDAL Python plugins')

    args = parser.parse_args()

    if args.pdal_driver_path:
        print_driver_path(args)
    elif args.pdal_plugin_path:
        print_plugin_path(args)
    else:
        print_version(args)


if __name__ == "__main__":
    main()
