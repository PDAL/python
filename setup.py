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
from distutils.core import setup, Extension
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
        import pdb;pdb.set_trace()


USE_CYTHON = True
try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.cpp'

logging.basicConfig()
log = logging.getLogger(__file__)

# python -W all setup.py ...
if 'all' in sys.warnoptions:
    log.level = logging.DEBUG


# Second try: use PDAL_CONFIG environment variable
if 'PDAL_CONFIG' in os.environ:
    pdal_config = os.environ['PDAL_CONFIG']
    log.debug('pdal_config: %s', pdal_config)
else:
    pdal_config = 'pdal-config'
    # in case of windows...
    if os.name in ['nt']:
        pdal_config += '.bat'


def get_pdal_config(option):
    '''Get configuration option from the `pdal-config` development utility

    This code was adapted from Shapely's geos-config stuff
    '''
    import subprocess
    pdal_config = globals().get('pdal_config')
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
    log.debug('%s %s: %r', pdal_config, option, result)
    return result

# Get the version from the pdal module
module_version = None
with open('pdal/__init__.py', 'r') as fp:
    for line in fp:
        if line.startswith("__version__"):
            module_version = Version(line.split("=")[1].strip().strip("\"'"))
            break

if not module_version:
    raise ValueError("Could not determine PDAL's version")

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
if pdal_config and "clean" not in sys.argv:

    PDAL_PLUGIN_DIR = get_pdal_config('--plugin-dir')
    PDAL_VERSION = Version(get_pdal_config('--version'))
    for item in get_pdal_config('--includes').split():
        if item.startswith("-I"):
            include_dirs.extend(item[2:].split(os.pathsep))

    for item in get_pdal_config('--libs').split():
        if item.startswith("-L"):
            library_dirs.extend(item[2:].split(os.pathsep))
        elif item.startswith("-l"):
            libraries.append(item[2:])

include_dirs.append(numpy.get_include())

if platform.system() == 'Darwin':
    extra_link_args.append('-Wl,-rpath,'+library_dirs[0])

if  PDAL_VERSION is not None \
    and PDAL_VERSION < Version('2.1.0'):
    raise Exception("PDAL version '%s' is not compatible with PDAL Python library version '%s'"%(PDAL_VERSION, module_version))


if os.name in ['nt']:
    if os.environ.get('CONDA_PREFIX'):
        prefix=os.path.expandvars('%CONDA_PREFIX%')
        library_dirs = ['%s\Library\lib' % prefix]

    libraries = ['pdalcpp','pdal_util','ws2_32']

    extra_compile_args = ['/DNOMINMAX',]

if 'linux' in sys.platform or 'linux2' in sys.platform or 'darwin' in sys.platform:
    extra_compile_args += ['-std=c++11', '-Wno-unknown-pragmas']
    if 'GCC' in sys.version:
        # try to ensure the ABI for Conda GCC 4.8
        if '4.8' in sys.version:
            extra_compile_args += ['-D_GLIBCXX_USE_CXX11_ABI=0']


# # This junk is here because the PDAL embedded environment needs the
# # Python library at compile time so it knows what to open. If the
# # Python environment was statically built (like Conda/OSX), we need to
# # do -undefined dynamic_lookup which the Python LDSHARED variable
# # gives us.
PYTHON_LIBRARY = os.path.join(sysconfig.get_config_var('LIBDIR'),
                              sysconfig.get_config_var('LDLIBRARY'))
# SHARED = sysconfig.get_config_var('Py_ENABLE_SHARED')
#
# # If we were build shared, just point to that. Otherwise,
# # point to the LDSHARED stuff and let dynamic_lookup find
# # it for us
# if not SHARED:
#     ldshared = ' '.join(sysconfig.get_config_var('LDSHARED').split(' ')[1:])
#     ldshared = ldshared.replace('-bundle','')
#     ldshared = [i for i in ldshared.split(' ') if i != '']

c = new_compiler()

for d in include_dirs:
    c.add_include_dir(d)
c.add_include_dir(get_python_inc())

c.add_library_dir(library_dirs[0])
c.add_library('pdalcpp')
c.add_library_dir(sysconfig.get_config_var('LIBDIR'))
PYLIB = sysconfig.get_config_var('LDLIBRARY').replace(c.dylib_lib_extension,'').replace('lib','')
c.add_library(PYLIB)
c.add_library('c++')

READER_FILENAME = c.dylib_lib_format % ('pdal_plugin_reader_numpy', c.dylib_lib_extension)
FILTER_FILENAME = c.dylib_lib_format % ('pdal_plugin_filter_python', c.dylib_lib_extension)

c.define_macro('PDAL_PYTHON_LIBRARY="%s"' % PYTHON_LIBRARY)

plang = c.compile(glob.glob('./pdal/plang/*.cpp'),
                  extra_preargs = extra_compile_args)

filter_objs = c.compile(glob.glob('./pdal/filters/*.cpp') ,
                        output_dir = temp_output_dir,
                        extra_preargs = extra_compile_args)

reader_objs = c.compile(glob.glob('./pdal/io/*.cpp') ,
                        output_dir = temp_output_dir,
                        extra_preargs = extra_compile_args)

filter_lib = c.link('shared_library', filter_objs + plang,
                     output_filename = FILTER_FILENAME,
                     output_dir = lib_output_dir,
                     extra_preargs = extra_link_args)

reader_lib = c.link('shared_library', reader_objs + plang,
                     output_filename = READER_FILENAME,
                     output_dir = lib_output_dir,
                     extra_preargs = extra_link_args)

if platform.system() == 'Darwin':
    import delocate

    def relocate(LIBRARY_NAME, library_output_dir):
        names = delocate.tools.get_install_names(os.path.join(library_output_dir, LIBRARY_NAME))
        inst_id = delocate.tools.get_install_id(os.path.join(library_output_dir, LIBRARY_NAME))
        set_id = delocate.tools.set_install_id(os.path.join(library_output_dir, LIBRARY_NAME), os.path.join('@rpath', LIBRARY_NAME))

    relocate(READER_FILENAME, lib_output_dir)
    relocate(FILTER_FILENAME, lib_output_dir)
    extra_link_args.append('-Wl,-rpath,'+library_dirs[0])

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
        'delocate ; platform_system=="Darwin"',
        'cython'],
)
output = setup(ext_modules=extensions, **setup_args)

