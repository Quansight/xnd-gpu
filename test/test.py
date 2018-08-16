import sys
sys.path.append('../xnd-gpu')

from xnd_gpu import xnd_gpu

import gpu_func
from xnd import xnd
from gumath import functions as fn
import gumath as gm
import numpy as np
import scipy.special as sp
from time import time
from pandas import DataFrame

binary_op = 'add', 'subtract', 'multiply', 'divide'
unary_op = 'fabs', 'exp', 'exp2', 'expm1', 'log', 'log2', 'log10', 'log1p', 'logb', 'sqrt', 'cbrt', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'erf', 'erfc', 'lgamma', 'tgamma', 'ceil', 'floor', 'trunc', 'round', 'nearbyint'
operations = {'binary': binary_op, 'unary': unary_op}

size = 10_000_000

in0 = np.random.uniform(0, 1, size=size).astype('float64')
in1 = np.random.uniform(0, 1, size=size).astype('float64')
dummy = np.empty_like(in0)

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

at_least = 10 # run at least 10 seconds

def run(op, args, kwargs={}, at_least=10):
    done = False
    n = 0
    t0 = time()
    while not done:
        for _ in range(100):
            op(*args, **kwargs)
            n += 1
        t1 = time()
        if t1 - t0 >= at_least:
            done = True
    return (t1 - t0) / n

for op_type in operations:
    for op_name in operations[op_type]:
        print(f'operation: {op_name}')

        # GPU
        op = gpu_func.__getattribute__(f'gpu_{op_name}')
        if op_type == 'binary':
            args = xgin0, xgin1, xgout
        else:
            args = xgin0, xgout
        xgt = run(op, args, at_least=at_least)
        #t0 = time()
        #xgout.gpu_copy_to_buffer(dummy) # get data from GPU back to CPU after the series of functions
        #t1 = time()
        #xgt += t1 - t0
        xgtime.append(xgt)
        print(f'GPU:   {xgt}')

        # CPU
        op = fn.__getattribute__(op_name)
        if op_type == 'binary':
            args = xin0, xin1
        else:
            args = xin0,
        xt = run(op, args, at_least=at_least)
        xtime.append(xt)
        print(f'CPU:   {xt}')

        # NumPy
        np_name = {'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan', 'asinh': 'arcsinh', 'acosh': 'arccosh', 'atanh': 'arctanh','nearbyint': 'rint', 'lgamma': 'gammaln', 'tgamma': 'gamma', 'logb': 'log2'}
        if op_name in np_name:
            op_name = np_name[op_name]
        package = None
        if op_name in np.__dir__():
            package = np
        elif op_name in sp.__dir__():
            package = sp
        if package is None:
            ntime.append(0)
            print(f'NumPy: NA')
        else:
            op = package.__getattribute__(op_name)
            if op_type == 'binary':
                args = in0, in1
            else:
                args = in0,
            if package == np:
                kwargs = {'out': dummy}
            else:
                kwargs = {}
            #nt = run(op, args, kwargs=kwargs, at_least=at_least)
            nt = run(op, args, at_least=at_least)
            ntime.append(nt)
            print(f'NumPy: {nt}')

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
fig.savefig('benchmark.png', bbox_inches='tight')
