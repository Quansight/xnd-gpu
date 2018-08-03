binary_op = 'add', 'subtract', 'multiply', 'divide'
unary_op = 'fabs','exp','exp2','expm1','log','log2','log10','log1p','logb','sqrt','cbrt','sin','cos','tan','asin','acos','atan','sinh','cosh','tanh','asinh','acosh','atanh','erf','erfc','lgamma','tgamma','ceil','floor','trunc','round','nearbyint'

binary_kernel = '''[KERNEL gpu_OP_NAME]
prototypes =
     void gpu_OP_NAME_float64_float64_float64(int   n, float64_t *  in0, float64_t *  in1, float64_t *  out);
description =
dimension = in0(n), in1(n), out(n)
input_arguments = in0, in1
inplace_arguments = out
hide_arguments = n = len(in0)

'''

unary_kernel = '''[KERNEL gpu_OP_NAME]
prototypes =
     void gpu_OP_NAME_float64_float64(int   n, float64_t *  in0, float64_t *  out);
description =
dimension = in0(n), out(n)
input_arguments = in0
inplace_arguments = out
hide_arguments = n = len(in0)

'''

kernels = '''[MODULE gpu_func]
typemaps =
     float32_t: float32
     float64_t: float64
     int: int32
includes =
     gpu_func.h
include_dirs =
sources =
     gpu_func.cu

libraries =

library_dirs =

header_code =
kinds = C
ellipses = none

'''

for op in binary_op:
    kernels += binary_kernel.replace('OP_NAME', op)
for op in unary_op:
    kernels += unary_kernel.replace('OP_NAME', op)

with open('gpu_func-kernels.cfg', 'w') as f:
    f.write(kernels)
