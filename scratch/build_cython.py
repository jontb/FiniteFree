import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    ext_modules=cythonize(
        Extension(
            "finitefree.utils.modular_fast",
            ["finitefree/utils/modular_fast.pyx"],
            include_dirs=[numpy.get_include()],
        ),
        language_level="3",
    ),
    script_args=["build_ext", "--inplace"],
)
