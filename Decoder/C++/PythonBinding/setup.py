from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

setup(ext_modules=[Extension("decoder_wrapper", 
                             ["decoder_wrapper.pyx", "../decoder.cpp"], language="c++",)],
      cmdclass = {'build_ext': build_ext})

