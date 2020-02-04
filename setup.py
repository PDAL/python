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
from setuptools import setup
from distutils.core import  Extension
from packaging.version import Version

from distutils.ccompiler import new_compiler
import distutils.unixccompiler
from distutils.sysconfig import get_python_inc
from distutils.command import build_ext
from distutils.util import get_platform
from distutils.command.install_data import install_data

class PDALInstallData(install_data):
    def run(self):
        install_data.run(self)


USE_CYTHON = True
try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.cpp'

PYVERS = sysconfig.get_config_var('py_version_nodot')
WINDOWS = False
if os.name in ['nt']:
    WINDOWS = True


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

include_dirs = []
library_dirs = []
libraries = []
extra_link_args = []
extra_compile_args = []

base = 'build'
plat_specifier = ".%s-%d.%d" % (get_platform(), *sys.version_info[:2])

lib_output_dir = os.path.join(base, 'lib' + plat_specifier)
temp_output_dir = os.path.join(base, 'temp' + plat_specifier)


PDAL_PLUGIN_DIR = None
PDAL_VERSION = None
if not WINDOWS:
    PDAL_PLUGIN_DIR = get_pdal_config('--plugin-dir')
    PDAL_VERSION = Version(get_pdal_config('--version'))
    for item in get_pdal_config('--includes').split():
        if item.startswith("-I"):
            include_dirs.extend(item[2:].split(os.pathsep))
    include_dirs.append(numpy.get_include())

    for item in get_pdal_config('--libs').split():
        if item.startswith("-L"):
            library_dirs.extend(item[2:].split(os.pathsep))
        elif item.startswith("-l"):
            libraries.append(item[2:])
    if 'linux' in sys.platform or \
       'linux2' in sys.platform or \
       'darwin' in sys.platform:
        extra_compile_args += ['-std=c++17', '-Wno-unknown-pragmas']
else:
    if os.environ.get('CONDA_PREFIX'):
        prefix=os.path.expandvars('%CONDA_PREFIX%')
        library_dirs = ['%s\Library\lib' % prefix]
        include_dirs = ['%s\Library\include' % prefix]

    libraries = ['pdalcpp','pdal_util','ws2_32']
    include_dirs.append(numpy.get_include())
    extra_compile_args = ['/DNOMINMAX',
                          '-D_CRT_SECURE_NO_WARNINGS=1',
                          '/wd4250',
                          '/wd4800']

if not PDAL_PLUGIN_DIR:
    try:
        PDAL_PLUGIN_DIR = os.environ['PDAL_PLUGIN_DIR']
    except KeyError:
        pass


if platform.system() == 'Darwin':
    extra_link_args.append('-Wl,-rpath,'+library_dirs[0])

if  PDAL_VERSION is not None \
    and PDAL_VERSION < Version('2.0.0'):
    raise Exception("PDAL version '%s' is not compatible with PDAL Python library version '%s'"%(PDAL_VERSION, module_version))


c = new_compiler()

extension = None
format = None
library_type = 'shared_library'
if WINDOWS:
    library_type = 'shared_object'
try:
    extension = c.dylib_lib_extension
except AttributeError:
    extension = c.shared_lib_extension
try:

    format = c.dylib_lib_format
except AttributeError:
    format = c.shared_lib_format

# # This junk is here because the PDAL embedded environment needs the
# # Python library at compile time so it knows what to open. If the
# # Python environment was statically built (like Conda/OSX), we need to
# # do -undefined dynamic_lookup which the Python LDSHARED variable
# # gives us.
PYTHON_LIB_DIR = sysconfig.get_config_var('LIBDIR')
PYTHON_LIBRARY_NAME = None
PYTHON_LIBRARY = None
PYTHON_INCLUDE_DIR = get_python_inc()
if WINDOWS:
    PYTHON_LIB_DIR = os.path.join(sysconfig.get_config_var("prefix"), "libs")
    PYTHON_LIBRARY_NAME = "python%s" % PYVERS
    PYTHON_LIBRARY = os.path.join(PYTHON_LIB_DIR, "python%s.lib" % PYVERS)
else:
    PYTHON_LIB_DIR = sysconfig.get_config_var('LIBDIR')
    PYTHON_LIBRARY_NAME = sysconfig.get_config_var('LDLIBRARY').replace(c.dylib_lib_extension,'').replace('lib','')
    PYTHON_LIBRARY = os.path.join(sysconfig.get_config_var('LIBDIR'),
                                  sysconfig.get_config_var('LDLIBRARY'))

SHARED = sysconfig.get_config_var('Py_ENABLE_SHARED')

library_dirs.append(PYTHON_LIB_DIR)
libraries.append(PYTHON_LIBRARY_NAME)
include_dirs.append(PYTHON_INCLUDE_DIR)

# # If we were build shared, just point to that. Otherwise,
# # point to the LDSHARED stuff and let dynamic_lookup find
# # it for us
if not SHARED:
    if not WINDOWS:
        ldshared = ' '.join(sysconfig.get_config_var('LDSHARED').split(' ')[1:])
        ldshared = ldshared.replace('-bundle','')
        ldshared = [i for i in ldshared.split(' ') if i != '']

for d in include_dirs:
    c.add_include_dir(d)
for d in library_dirs:
    c.add_library_dir(d)
for d in libraries:
    c.add_library(d)

if not WINDOWS:
    c.add_library('c++')
else:
    extra_compile_args+=['-DPDAL_DLL_EXPORT=1']


READER_FILENAME = format % ('pdal_plugin_reader_numpy', extension)
FILTER_FILENAME = format % ('pdal_plugin_filter_python', extension)

if PYTHON_LIBRARY:
    c.define_macro('PDAL_PYTHON_LIBRARY="%s"' % PYTHON_LIBRARY)

plang = c.compile(glob.glob('./pdal/plang/*.cpp'),
                  extra_preargs = extra_compile_args)

filter_objs = c.compile(glob.glob('./pdal/filters/*.cpp') ,
                        output_dir = temp_output_dir,
                        extra_preargs = extra_compile_args)

reader_objs = c.compile(glob.glob('./pdal/io/*.cpp') ,
                        output_dir = temp_output_dir,
                        extra_preargs = extra_compile_args)

filter_lib = c.link(library_type, filter_objs + plang,
                     output_filename = FILTER_FILENAME,
                     output_dir = lib_output_dir,
                     extra_preargs = extra_link_args)

reader_lib = c.link(library_type, reader_objs + plang,
                     output_filename = READER_FILENAME,
                     output_dir = lib_output_dir,
                     extra_preargs = extra_link_args)

if platform.system() == 'Darwin':
    import delocate

    def relocate(LIBRARY_NAME, library_output_dir):
        LIB = os.path.join(library_output_dir, LIBRARY_NAME)
        names = delocate.tools.get_install_names(LIB)
        inst_id = delocate.tools.get_install_id(LIB)
        set_id = delocate.tools.set_install_id(LIB, os.path.join('@rpath', LIBRARY_NAME))

    relocate(READER_FILENAME, lib_output_dir)
    relocate(FILTER_FILENAME, lib_output_dir)

extensions = []
extension_sources=['pdal/libpdalpython'+ext, "pdal/PyPipeline.cpp", "pdal/PyArray.cpp" ]
extension = Extension("*",
                       extension_sources,
                       include_dirs = include_dirs,
                       library_dirs = library_dirs,
                       extra_compile_args = extra_compile_args,
                       libraries = libraries,
                       extra_link_args = extra_link_args,)

if USE_CYTHON and "clean" not in sys.argv:
    from Cython.Build import cythonize
    extensions = cythonize([extension], compiler_directives={'language_level':3})

DATA_FILES = None
if PDAL_PLUGIN_DIR:
    libs = [os.path.join(lib_output_dir,READER_FILENAME),
            os.path.join(lib_output_dir,FILTER_FILENAME)]
    DATA_FILES          = [(PDAL_PLUGIN_DIR, libs)]

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
    test_suite          = 'test',
    packages            = [
        'pdal',
    ],
    data_files=DATA_FILES,
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
    cmdclass           = {'install_data': PDALInstallData},
    install_requires   = [
        'numpy',
        'packaging',
        'delocate ; platform_system=="darwin"',
        'cython'],
    setup_requires   = [
        'numpy',
        'packaging',
        'delocate ; platform_system=="darwin"',
        'cython'],
)
output = setup(ext_modules=extensions, **setup_args)

