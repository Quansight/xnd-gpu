GPU can be faked (CPU mode).

GPU: nvcc -o libgpu_mem.so --compiler-options "-fPIC" --shared gpu_mem.cu
CPU: gcc -o libgpu_mem.so -fPIC --shared gpu_mem.c

xnd_tools kernel gpu_func-kernels.cfg
xnd_tools module gpu_func-kernels.cfg

GPU: nvcc --compiler-options "-fPIC" -c gpu_func.cu
CPU: gcc -fPIC -c gpu_func.c
export SITE_PACKAGES=`python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"`
gcc -fPIC -c gpu_func-kernels.c -c $SITE_PACKAGES/xndtools/kernel_generator/xndtools.c -I$SITE_PACKAGES/xndtools/kernel_generator -I$SITE_PACKAGES/xnd -I$SITE_PACKAGES/ndtypes -I$SITE_PACKAGES/gumath
ar rcs libgpu_func-kernels.a gpu_func.o gpu_func-kernels.o xndtools.o

rm -rf build
GPU: python setup.py install
CPU: python setup_cpu.py install
