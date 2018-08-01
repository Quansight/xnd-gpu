from distutils.core import setup, Extension
from distutils.sysconfig import get_python_lib

site_packages = get_python_lib()
lib_dirs = [f'{site_packages}/{i}' for i in ['ndtypes', 'gumath', 'xnd']]

module1 = Extension('gpu_func',
                    include_dirs = lib_dirs,
                    libraries = ['gpu_func-kernels', 'ndtypes','gumath', 'xnd', 'cudart', 'stdc++'],
                    library_dirs = ['.', '/usr/local/cuda-9.2/lib64'] + lib_dirs,
                    sources = ['gpu_func-python.c'])

setup (name = 'gpu_func',
       version = '1.0',
       description = 'This is a gumath kernel extension that provides GPU support for a number of binary and unary operations',
       ext_modules = [module1])
