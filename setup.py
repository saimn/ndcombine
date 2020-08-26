import numpy as np
import os
from setuptools import Extension, find_packages, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "*",
        ["ndcombine/*.pyx"],
        include_dirs=[np.get_include()],
        # libraries=[...],
        # library_dirs=[...]
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]

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

ext_modules = cythonize(ext_modules,
                        compiler_directives=compiler_directives,
                        gdb_debug=gdb_debug)

setup(
    name='ndcombine',
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
    ext_modules=cythonize(ext_modules),
)
