from setuptools import Extension, setup
from Cython.Build import cythonize

setup(ext_modules = cythonize(Extension(
           "decoder_wrapper",
           sources=["decoder_wrapper.pyx","../beam_search.cpp","../decoder.cpp","../fst.cpp","../helpers.cpp","../lattice.cpp"],  # additional source file(s)
           language="c++",
           libraries=["fst"],
      ), compiler_directives={'language_level' : "3"}))
