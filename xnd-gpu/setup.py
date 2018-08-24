from distutils.core import setup, Extension
from distutils.sysconfig import get_python_lib
import os

site_packages = get_python_lib()
xnd_dirs = [f'{site_packages}/{i}' for i in ['ndtypes', 'gumath', 'xnd']]
conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    xnd_dirs.append(f'{conda_prefix}/include')
cuda_dir = []
cudadir = os.environ.get('CUDADIR')
if cudadir:
    cuda_dir.append(cudadir)

module1 = Extension('gpu_func',
                    include_dirs = xnd_dirs,
                    libraries = ['gpu_func-kernels', 'ndtypes','gumath', 'xnd', 'cudart', 'stdc++'],
                    library_dirs = ['.'] + xnd_dirs + cuda_dir,
                    sources = ['gpu_func-python.c'])

setup (name = 'gpu_func',
       version = '1.0',
       description = 'This is a gumath kernel extension that provides GPU support for a number of binary and unary operations',
       ext_modules = [module1])
