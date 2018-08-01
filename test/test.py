from xnd_gpu import xnd_gpu, gpu_synchro
from gpu_func import gpu_add
from xnd import xnd
from gumath import functions as fn
import gumath as gm
import numpy as np
from time import time

times = 1_000
size = 1_000_000
shape = (size,)

a = np.arange(size, dtype='float32').reshape(shape)
b = np.ones_like(a)

# GPU

x = xnd_gpu(from_buffer=a)
y = xnd_gpu(from_buffer=b)

r = xnd_gpu(empty_like=a)

t0 = time()

for i in range(times):
    gpu_add(x, y, r)
gpu_synchro()

t1 = time()

print(f'GPU took {t1 - t0} seconds.')

# CPU

x = xnd.from_buffer(a)
y = xnd.from_buffer(b)

t0 = time()

for i in range(times):
    xr = fn.add(x, y)

t1 = time()

print(f'CPU took {t1 - t0} seconds.')

assert (r == xr)
print(r)
