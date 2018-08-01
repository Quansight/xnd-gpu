SITE_PACKAGES ?= `python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"`
CC = gcc
CFLAGS = -I$(SITE_PACKAGES)/xndtools/kernel_generator -I$(SITE_PACKAGES)/xnd -I$(SITE_PACKAGES)/ndtypes -I$(SITE_PACKAGES)/gumath

cpu gpu: libgpu_func-kernels.a gpu_func-python.c libgpu_mem.so
	rm -rf build
	python setup.py install
	if [ $(MAKECMDGOALS) = cpu ]; then \
		echo "Built for fake GPU"; \
	else \
		echo "Built for real GPU"; \
	fi

libgpu_mem.so: gpu_mem.cu gpu_mem.c
	if [ $(MAKECMDGOALS) = cpu ]; then \
		$(CC) -o libgpu_mem.so -fPIC --shared gpu_mem.c; \
	else \
		nvcc -o libgpu_mem.so --compiler-options "-fPIC" --shared gpu_mem.cu; \
	fi
	cp libgpu_mem.so $(SITE_PACKAGES)

gpu_func-kernels.c: gpu_func-kernels.cfg
	xnd_tools kernel gpu_func-kernels.cfg

gpu_func-python.c: gpu_func-kernels.c
	xnd_tools module gpu_func-kernels.cfg

gpu_func.o: gpu_func.cu
	if [ $(MAKECMDGOALS) = cpu ]; then \
		$(CC) -fPIC -c gpu_func.c; \
	else \
		nvcc --compiler-options "-fPIC" -c gpu_func.cu; \
	fi

gpu_func-kernels.o: gpu_func-kernels.c
	$(CC) -fPIC -c gpu_func-kernels.c $(CFLAGS)

xndtools.o: $(SITE_PACKAGES)/xndtools/kernel_generator/xndtools.c
	$(CC) -fPIC -c $(SITE_PACKAGES)/xndtools/kernel_generator/xndtools.c $(CFLAGS)

libgpu_func-kernels.a: gpu_func.o gpu_func-kernels.o xndtools.o
	ar rcs libgpu_func-kernels.a gpu_func.o gpu_func-kernels.o xndtools.o

clean:
	rm -rf build *.so *.a *.o
