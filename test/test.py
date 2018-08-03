import sys
sys.path.append('../xnd-gpu')

from xnd_gpu import xnd_gpu, gpu_synchro

import gpu_func
from xnd import xnd
from gumath import functions as fn
import gumath as gm
import numpy as np
from time import time
from pandas import DataFrame

binary_op = 'add', 'subtract', 'multiply', 'divide'
unary_op = 'fabs','exp','exp2','expm1','log','log2','log10','log1p','logb','sqrt','cbrt','sin','cos','tan','asin','acos','atan','sinh','cosh','tanh','asinh','acosh','atanh','erf','erfc','lgamma','tgamma','ceil','floor','trunc','round','nearbyint'
operations = {'binary': binary_op, 'unary': unary_op}

times = 100
size = 1_000_000

in0 = np.random.uniform(0, 1, size=size) #, dtype='float64')
in1 = np.random.uniform(0, 1, size=size) #, dtype='float64')

# binary operations

xgin0 = xnd_gpu(from_buffer=in0)
xgin1 = xnd_gpu(from_buffer=in1)

xgout = xnd_gpu(empty_like=in0)

xin0 = xnd.from_buffer(in0)
xin1 = xnd.from_buffer(in1)

index = []
xgtime = []
xtime = []
ntime = []

for op_type in operations:
    for op_name in operations[op_type]:
        print(f'operation: {op_name}')

        # GPU
        op = gpu_func.__getattribute__(f'gpu_{op_name}')
        if op_type == 'binary':
            args = xgin0, xgin1, xgout
        else:
            args = xgin0, xgout
        t0 = time()
        for i in range(times):
            op(*args)
        gpu_synchro()
        t1 = time()
        xgt = t1 - t0
        xgtime.append(xgt)
        print(f'GPU:   {xgt}')

        # CPU
        op = fn.__getattribute__(op_name)
        if op_type == 'binary':
            args = xin0, xin1
        else:
            args = xin0,
        t0 = time()
        for i in range(times):
            xout = op(*args)
        t1 = time()
        xt = t1 - t0
        xtime.append(xt)
        print(f'CPU:   {xt}')

        # NumPy
        np_name = {'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan', 'asinh': 'arcsinh', 'acosh': 'arccosh', 'atanh': 'arctanh','nearbyint': 'rint'}
        if op_name in np_name:
            op_name = np_name[op_name]
        if op_name in np.__dir__():
            op = np.__getattribute__(op_name)
            if op_type == 'binary':
                args = in0, in1
            else:
                args = in0,
            t0 = time()
            for i in range(times):
                out = op(*args)
            t1 = time()
            nt = t1 - t0
            ntime.append(nt)
            print(f'NumPy: {nt}')
        else:
            ntime.append(0)
            print(f'NumPy: NA')

        print()
        index.append(op_name)

        #print(gr)
        #assert (r == gr)
df = DataFrame({'XND-GPU': xgtime, 'XND': xtime, 'NumPy': ntime}, index=index)
df = (1 / df).replace(np.inf ,0)
df = df.div(df.max(axis=1), axis=0)
df.to_pickle('benchmark.pkl')
ax = df.plot.bar(title='Speed (1 is fastest)', figsize=(15, 5))
fig = ax.get_figure()
fig.savefig('benchmark.png')
