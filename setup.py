#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# NOTE: The configuration for the package, including the name, version, and
# other information are set in the setup.cfg file.

import os
import sys

from setuptools import Extension, setup

# First provide helpful messages if contributors try and run legacy commands
# for tests or docs.

TEST_HELP = """
Note: running tests is no longer done using 'python setup.py test'. Instead
you will need to run:

    tox -e test

If you don't already have tox installed, you can install it with:

    pip install tox

If you only want to run part of the test suite, you can also use pytest
directly with::

    pip install -e .[test]
    pytest

For more information, see:

  http://docs.astropy.org/en/latest/development/testguide.html#running-tests
"""

if 'test' in sys.argv:
    print(TEST_HELP)
    sys.exit(1)

DOCS_HELP = """
Note: building the documentation is no longer done using
'python setup.py build_docs'. Instead you will need to run:

    tox -e build_docs

If you don't already have tox installed, you can install it with:

    pip install tox

You can also build the documentation with Sphinx directly using::

    pip install -e .[docs]
    cd docs
    make html

For more information, see:

  http://docs.astropy.org/en/latest/install.html#builddocs
"""

if 'build_docs' in sys.argv or 'build_sphinx' in sys.argv:
    print(DOCS_HELP)
    sys.exit(1)

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

# Import this later to allow checking deprecated options before
import numpy as np  # noqa
from Cython.Build import cythonize  # noqa
from extension_helpers import add_openmp_flags_if_available  # noqa

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

setup(ext_modules=ext_modules,
      use_scm_version={'write_to': os.path.join('ndcombine', 'version.py'),
                       'write_to_template': VERSION_TEMPLATE})
