SITE_PACKAGES ?= $(shell python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
CC = gcc
CFLAGS = -I$(SITE_PACKAGES)/xndtools/kernel_generator -I$(SITE_PACKAGES)/xnd -I$(SITE_PACKAGES)/ndtypes -I$(SITE_PACKAGES)/gumath
ifeq ($(MAKECMDGOALS),cpu)
	GPU = fake
else
	GPU = real
endif

cpu gpu: libgpu_func-kernels.a gpu_func-python.c libgpu_mem.so
	cp libgpu_mem.so $(SITE_PACKAGES)
	rm -rf build
	python setup.py install
	@echo Built for $(GPU) GPU

ifeq ($(MAKECMDGOALS),cpu)
libgpu_mem.so: gpu_mem.c
	$(CC) -o libgpu_mem.so -fPIC --shared gpu_mem.c
else
libgpu_mem.so: gpu_mem.cu
	nvcc -o libgpu_mem.so --compiler-options "-fPIC" --shared gpu_mem.cu
endif

gpu_func-kernels.cfg:
	python write_cfg.py

gpu_func-kernels.c: gpu_func-kernels.cfg
	xnd_tools kernel gpu_func-kernels.cfg

gpu_func-python.c: gpu_func-kernels.c
	xnd_tools module gpu_func-kernels.cfg

ifeq ($(MAKECMDGOALS),cpu)
gpu_func.o: gpu_func.c
	$(CC) -fPIC -c gpu_func.c
else
gpu_func.o: gpu_func.cu
	nvcc --compiler-options "-fPIC" -c gpu_func.cu
endif

gpu_func-kernels.o: gpu_func-kernels.c
	$(CC) -fPIC -c gpu_func-kernels.c $(CFLAGS)

xndtools.o: $(SITE_PACKAGES)/xndtools/kernel_generator/xndtools.c
	$(CC) -fPIC -c $(SITE_PACKAGES)/xndtools/kernel_generator/xndtools.c $(CFLAGS)

libgpu_func-kernels.a: gpu_func.o gpu_func-kernels.o xndtools.o
	ar rcs libgpu_func-kernels.a gpu_func.o gpu_func-kernels.o xndtools.o

clean:
	rm -rf build *.so *.a *.o
