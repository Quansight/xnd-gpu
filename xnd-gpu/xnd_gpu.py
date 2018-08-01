import ctypes
import numpy as np
from xnd import xnd
from functools import reduce
import operator
import types

gpu_mem = ctypes.CDLL('libgpu_mem.so')

gpu_mem.new_array_float32.restype = ctypes.POINTER(ctypes.c_float)
gpu_mem.new_array_float64.restype = ctypes.POINTER(ctypes.c_double)

gpu_mem.del_array_float32.argtypes = [ctypes.POINTER(ctypes.c_float), ]
gpu_mem.del_array_float64.argtypes = [ctypes.POINTER(ctypes.c_double), ]

gpu_mem.copy_array_float32.argtype = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ]
gpu_mem.copy_array_float64.argtype = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ]

def copy_from_buffer(self, a):
    if self.gpu_dtype == 'float32':
        gpu_mem.copy_array_float32(self.gpu_addr, ctypes.c_void_p(a.__array_interface__['data'][0]), self.gpu_size)
    elif self.gpu_dtype == 'float64':
        gpu_mem.copy_array_float64(self.gpu_addr, ctypes.c_void_p(a.__array_interface__['data'][0]), self.gpu_size)

def copy_to_buffer(self, a):
    if self.gpu_dtype == 'float32':
        gpu_mem.copy_array_float32(ctypes.c_void_p(a.__array_interface__['data'][0]), self.gpu_addr, self.gpu_size)
    elif self.gpu_dtype == 'float64':
        gpu_mem.copy_array_float64(ctypes.c_void_p(a.__array_interface__['data'][0]), self.gpu_addr, self.gpu_size)

def delete(self):
    '''Delete the data allocated on the GPU.
    '''
    if self.gpu_addr is not None:
        if self.gpu_dtype == 'float32':
            gpu_mem.del_array_float32(self.gpu_addr)
        elif self.gpu_dtype == 'float64':
            gpu_mem.del_array_float64(self.gpu_addr)

def gpu_synchro():
    gpu_mem.synchro()

def new(shape, dtype='float32'):
    '''Allocate on the GPU the necessary memory to hold an array of given
    shape and dtype.
    Return an XND container whose data points to that memory, augmented with
    some GPU attributes.
    '''
    size = reduce(operator.mul, shape, 1)
    if dtype == 'float32':
        addr = gpu_mem.new_array_float32(size)
    elif dtype == 'float64':
        addr = gpu_mem.new_array_float64(size)
    a = np.ctypeslib.as_array(addr, shape=shape)
    x = xnd.from_buffer(a)
    x.gpu_addr = addr
    x.gpu_dtype = dtype
    x.gpu_shape = shape
    x.gpu_size = size
    x.gpu_copy_from_buffer = types.MethodType(copy_from_buffer, x)
    x.gpu_copy_to_buffer = types.MethodType(copy_to_buffer, x)
    x.gpu_delete = types.MethodType(delete, x)
    return x

def xnd_gpu(shape=None, dtype=None, from_buffer=None, empty_like=None):
    if not any([shape is None, dtype is None]):
        return new(shape=shape, dtype=dtype)
    elif empty_like is not None:
        a = empty_like
        return new(shape=a.shape, dtype=a.dtype)
    elif from_buffer is not None:
        a = from_buffer
        x = new(shape=a.shape, dtype=str(a.dtype))
        x.gpu_copy_from_buffer(a)
        return x
