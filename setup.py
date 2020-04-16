#!/usr/bin/env python

# Stolen from Shapely's setup.py
# Two environment variables influence this script.
#
# PDAL_LIBRARY_PATH: a path to a PDAL C++ shared library.
#
# PDAL_CONFIG: the path to a pdal-config program that points to PDAL version,
# headers, and libraries.
#
# NB: within this setup scripts, software versions are evaluated according
# to https://www.python.org/dev/peps/pep-0440/.

import logging
import os
import platform
import sys
import numpy
import glob
import sysconfig
from skbuild import setup
from packaging.version import Version



def get_pdal_config(option):
    '''Get configuration option from the `pdal-config` development utility

    This code was adapted from Shapely's geos-config stuff
    '''
    import subprocess
    pdal_config = os.environ.get('PDAL_CONFIG','pdal-config')
    if not pdal_config or not isinstance(pdal_config, str):
        raise OSError('Path to pdal-config is not set')
    try:
        stdout, stderr = subprocess.Popen(
            [pdal_config, option],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    except OSError as ex:
        # e.g., [Errno 2] No such file or directory
        raise OSError(
            'Could not find pdal-config %r: %s' % (pdal_config, ex))
    if stderr and not stdout:
        raise ValueError(stderr.strip())
    if sys.version_info[0] >= 3:
        result = stdout.decode('ascii').strip()
    else:
        result = stdout.strip()
    return result

# Get the version from the pdal module
module_version = None
with open('pdal/__init__.py', 'r') as fp:
    for line in fp:
        if line.startswith("__version__"):
            module_version = Version(line.split("=")[1].strip().strip("\"'"))
            break

if not module_version:
    raise ValueError("Could not determine Python package version")

# Handle UTF-8 encoding of certain text files.
open_kwds = {}
if sys.version_info >= (3,):
    open_kwds['encoding'] = 'utf-8'

with open('README.rst', 'r', **open_kwds) as fp:
    readme = fp.read()

with open('CHANGES.txt', 'r', **open_kwds) as fp:
    changes = fp.read()

long_description = readme + '\n\n' +  changes


setup_args = dict(
    name                = 'PDAL',
    version             = str(module_version),
    requires            = ['Python (>=3.0)', 'Numpy'],
    description         = 'Point cloud data processing',
    license             = 'BSD',
    keywords            = 'point cloud spatial',
    author              = 'Howard Butler',
    author_email        = 'howard@hobu.co',
    maintainer          = 'Howard Butler',
    maintainer_email    = 'howard@hobu.co',
    url                 = 'https://pdal.io',
    long_description    = long_description,
    long_description_content_type = 'text/x-rst',
    test_suite          = 'test',
    cmake_source_dir    = 'pdal',
    packages            = [
        'pdal',
    ],
    classifiers         = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: GIS',
    ],

)
output = setup(**setup_args)

