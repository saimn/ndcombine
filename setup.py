#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# NOTE: The configuration for the package, including the name, version, and
# other information are set in the setup.cfg file.

import os


import numpy as np
from Cython.Build import cythonize
from extension_helpers import add_openmp_flags_if_available
from setuptools import Extension, find_packages, setup

extension = Extension(
    "*",
    ["ndcombine/*.pyx"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

add_openmp_flags_if_available(extension)

compiler_directives = {}

if os.getenv('COVERAGE'):
    print('Adding linetrace directive')
    compiler_directives['profile'] = True
    compiler_directives['linetrace'] = True
    compiler_directives['emit_code_comments'] = True
    os.environ['CFLAGS'] = ('-g -DCYTHON_TRACE_NOGIL=1 --coverage '
                            '-fno-inline-functions -O0')
    gdb_debug = True
else:
    gdb_debug = False

ext_modules = cythonize([extension],
                        compiler_directives=compiler_directives,
                        gdb_debug=gdb_debug)

VERSION_TEMPLATE = """
# Note that we need to fall back to the hard-coded version if either
# setuptools_scm can't be imported or setuptools_scm can't determine the
# version, so we catch the generic 'Exception'.
try:
    from setuptools_scm import get_version
    version = get_version(root='..', relative_to=__file__)
except Exception:
    version = '{version}'
""".lstrip()

setup(
    name='ndcombine',
    description='Fast ND arrays combination',
    author='Simon Conseil',
    url='https://github.com/saimn/ndcombine',
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
    ext_modules=ext_modules,
    python_requires='>=3.6',
    install_requires=['numpy', 'astropy'],
    extras_require={
        'tests': ['pytest'],
    },
    use_scm_version={'write_to': os.path.join('ndcombine', 'version.py'),
                     'write_to_template': VERSION_TEMPLATE},
)
