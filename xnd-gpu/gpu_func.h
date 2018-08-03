#ifndef gpu_func_h
#define gpu_func_h

#include <stdint.h>

#ifdef FAKE_GPU
#include <math.h>
#endif

typedef float float32_t;
typedef double float64_t;

#ifdef __cplusplus
    #define GPU_BINARY_PROTO(func, t0, t1, t2) extern "C" void gpu_##func##_##t0##_##t1##_##t2(int n, t0##_t* in0, t1##_t* in1, t2##_t* out);
#else
    #define GPU_BINARY_PROTO(func, t0, t1, t2) void gpu_##func##_##t0##_##t1##_##t2(int n, t0##_t* in0, t1##_t* in1, t2##_t* out);
#endif

#ifdef __cplusplus
    #define GPU_UNARY_PROTO(func, t0, t1) extern "C" void gpu_##func##_##t0##_##t1(int n, t0##_t* in0, t1##_t* out);
#else
    #define GPU_UNARY_PROTO(func, t0, t1) void gpu_##func##_##t0##_##t1(int n, t0##_t* in0, t1##_t* out);
#endif

#define GPU_ALL_BINARY_PROTO(name)              \
    GPU_BINARY_PROTO(name, int8, int8, int8)          \
    GPU_BINARY_PROTO(name, int8, int16, int16)        \
    GPU_BINARY_PROTO(name, int8, int32, int32)        \
    GPU_BINARY_PROTO(name, int8, int64, int64)        \
    GPU_BINARY_PROTO(name, int8, uint8, int16)        \
    GPU_BINARY_PROTO(name, int8, uint16, int32)       \
    GPU_BINARY_PROTO(name, int8, uint32, int64)       \
    GPU_BINARY_PROTO(name, int8, float32, float32)    \
    GPU_BINARY_PROTO(name, int8, float64, float64)    \
                                                \
    GPU_BINARY_PROTO(name, int16, int8, int16)        \
    GPU_BINARY_PROTO(name, int16, int16, int16)       \
    GPU_BINARY_PROTO(name, int16, int32, int32)       \
    GPU_BINARY_PROTO(name, int16, int64, int64)       \
    GPU_BINARY_PROTO(name, int16, uint8, int16)       \
    GPU_BINARY_PROTO(name, int16, uint16, int32)      \
    GPU_BINARY_PROTO(name, int16, uint32, int64)      \
    GPU_BINARY_PROTO(name, int16, float32, float32)   \
    GPU_BINARY_PROTO(name, int16, float64, float64)   \
                                                \
    GPU_BINARY_PROTO(name, int32, int8, int32)        \
    GPU_BINARY_PROTO(name, int32, int16, int32)       \
    GPU_BINARY_PROTO(name, int32, int32, int32)       \
    GPU_BINARY_PROTO(name, int32, int64, int64)       \
    GPU_BINARY_PROTO(name, int32, uint8, int32)       \
    GPU_BINARY_PROTO(name, int32, uint16, int32)      \
    GPU_BINARY_PROTO(name, int32, uint32, int64)      \
    GPU_BINARY_PROTO(name, int32, float64, float64)   \
                                                \
    GPU_BINARY_PROTO(name, int64, int8, int64)        \
    GPU_BINARY_PROTO(name, int64, int16, int64)       \
    GPU_BINARY_PROTO(name, int64, int32, int64)       \
    GPU_BINARY_PROTO(name, int64, int64, int64)       \
    GPU_BINARY_PROTO(name, int64, uint8, int64)       \
    GPU_BINARY_PROTO(name, int64, uint16, int64)      \
    GPU_BINARY_PROTO(name, int64, uint32, int64)      \
                                                \
    GPU_BINARY_PROTO(name, uint8, int8, int16)        \
    GPU_BINARY_PROTO(name, uint8, int16, int16)       \
    GPU_BINARY_PROTO(name, uint8, int32, int32)       \
    GPU_BINARY_PROTO(name, uint8, int64, int64)       \
    GPU_BINARY_PROTO(name, uint8, uint8, uint8)       \
    GPU_BINARY_PROTO(name, uint8, uint16, uint16)     \
    GPU_BINARY_PROTO(name, uint8, uint32, uint32)     \
    GPU_BINARY_PROTO(name, uint8, uint64, uint64)     \
    GPU_BINARY_PROTO(name, uint8, float32, float32)   \
    GPU_BINARY_PROTO(name, uint8, float64, float64)   \
                                                \
    GPU_BINARY_PROTO(name, uint16, int8, int32)       \
    GPU_BINARY_PROTO(name, uint16, int16, int32)      \
    GPU_BINARY_PROTO(name, uint16, int32, int32)      \
    GPU_BINARY_PROTO(name, uint16, int64, int64)      \
    GPU_BINARY_PROTO(name, uint16, uint8, uint16)     \
    GPU_BINARY_PROTO(name, uint16, uint16, uint16)    \
    GPU_BINARY_PROTO(name, uint16, uint32, uint32)    \
    GPU_BINARY_PROTO(name, uint16, uint64, uint64)    \
    GPU_BINARY_PROTO(name, uint16, float32, float32)  \
    GPU_BINARY_PROTO(name, uint16, float64, float64)  \
                                                \
    GPU_BINARY_PROTO(name, uint32, int8, int64)       \
    GPU_BINARY_PROTO(name, uint32, int16, int64)      \
    GPU_BINARY_PROTO(name, uint32, int32, int64)      \
    GPU_BINARY_PROTO(name, uint32, int64, int64)      \
    GPU_BINARY_PROTO(name, uint32, uint8, uint32)     \
    GPU_BINARY_PROTO(name, uint32, uint16, uint16)    \
    GPU_BINARY_PROTO(name, uint32, uint32, uint32)    \
    GPU_BINARY_PROTO(name, uint32, uint64, uint64)    \
    GPU_BINARY_PROTO(name, uint32, float64, float64)  \
                                                \
    GPU_BINARY_PROTO(name, uint64, uint8, uint64)     \
    GPU_BINARY_PROTO(name, uint64, uint16, uint64)    \
    GPU_BINARY_PROTO(name, uint64, uint32, uint64)    \
    GPU_BINARY_PROTO(name, uint64, uint64, uint64)    \
                                                \
    GPU_BINARY_PROTO(name, float32, int8, float32)    \
    GPU_BINARY_PROTO(name, float32, int16, float32)   \
    GPU_BINARY_PROTO(name, float32, uint8, float32)   \
    GPU_BINARY_PROTO(name, float32, uint16, float32)  \
    GPU_BINARY_PROTO(name, float32, float32, float32) \
    GPU_BINARY_PROTO(name, float32, float64, float64) \
                                                \
    GPU_BINARY_PROTO(name, float64, int8, float64)    \
    GPU_BINARY_PROTO(name, float64, int16, float64)   \
    GPU_BINARY_PROTO(name, float64, int32, float64)   \
    GPU_BINARY_PROTO(name, float64, uint8, float64)   \
    GPU_BINARY_PROTO(name, float64, uint16, float64)  \
    GPU_BINARY_PROTO(name, float64, uint32, float64)  \
    GPU_BINARY_PROTO(name, float64, float32, float64) \
    GPU_BINARY_PROTO(name, float64, float64, float64)

#define GPU_ALL_UNARY_FLOAT_PROTO(name)        \
    GPU_UNARY_PROTO(name##f, int8, float32)    \
    GPU_UNARY_PROTO(name##f, int16, float32)   \
    GPU_UNARY_PROTO(name##f, uint8, float32)   \
    GPU_UNARY_PROTO(name##f, uint16, float32)  \
    GPU_UNARY_PROTO(name##f, float32, float32) \
/*  GPU_UNARY_PROTO(name, int32, float64)      \
    GPU_UNARY_PROTO(name, uint32, float64)     \
*/  GPU_UNARY_PROTO(name, float64, float64)

GPU_ALL_BINARY_PROTO(add)
GPU_ALL_BINARY_PROTO(subtract)
GPU_ALL_BINARY_PROTO(multiply)
GPU_ALL_BINARY_PROTO(divide)

GPU_ALL_UNARY_FLOAT_PROTO(fabs)
GPU_ALL_UNARY_FLOAT_PROTO(exp)
GPU_ALL_UNARY_FLOAT_PROTO(exp2)
GPU_ALL_UNARY_FLOAT_PROTO(expm1)
GPU_ALL_UNARY_FLOAT_PROTO(log)
GPU_ALL_UNARY_FLOAT_PROTO(log2)
GPU_ALL_UNARY_FLOAT_PROTO(log10)
GPU_ALL_UNARY_FLOAT_PROTO(log1p)
GPU_ALL_UNARY_FLOAT_PROTO(logb)
GPU_ALL_UNARY_FLOAT_PROTO(sqrt)
GPU_ALL_UNARY_FLOAT_PROTO(cbrt)
GPU_ALL_UNARY_FLOAT_PROTO(sin)
GPU_ALL_UNARY_FLOAT_PROTO(cos)
GPU_ALL_UNARY_FLOAT_PROTO(tan)
GPU_ALL_UNARY_FLOAT_PROTO(asin)
GPU_ALL_UNARY_FLOAT_PROTO(acos)
GPU_ALL_UNARY_FLOAT_PROTO(atan)
GPU_ALL_UNARY_FLOAT_PROTO(sinh)
GPU_ALL_UNARY_FLOAT_PROTO(cosh)
GPU_ALL_UNARY_FLOAT_PROTO(tanh)
GPU_ALL_UNARY_FLOAT_PROTO(asinh)
GPU_ALL_UNARY_FLOAT_PROTO(acosh)
GPU_ALL_UNARY_FLOAT_PROTO(atanh)
GPU_ALL_UNARY_FLOAT_PROTO(erf)
GPU_ALL_UNARY_FLOAT_PROTO(erfc)
GPU_ALL_UNARY_FLOAT_PROTO(lgamma)
GPU_ALL_UNARY_FLOAT_PROTO(tgamma)
GPU_ALL_UNARY_FLOAT_PROTO(ceil)
GPU_ALL_UNARY_FLOAT_PROTO(floor)
GPU_ALL_UNARY_FLOAT_PROTO(trunc)
GPU_ALL_UNARY_FLOAT_PROTO(round)
GPU_ALL_UNARY_FLOAT_PROTO(nearbyint)

#ifdef FAKE_GPU

#define GPU_BINARY(func, t0, t1, t2)                                                \
void gpu_##func##_##t0##_##t1##_##t2(int n, t0##_t* in0, t1##_t* in1, t2##_t* out)  \
{                                                                                   \
/*  for (int i = 0; i < n; i ++)                                                    \
        out[i] = func(in0[i], in1[i]);*/                                            \
    int i;                                                                          \
    for (i = 0; i < n-7; i += 8) {                                                  \
        out[i] = func(in0[i], in1[i]);                                              \
        out[i+1] = func(in0[i+1], in1[i+1]);                                        \
        out[i+2] = func(in0[i+2], in1[i+2]);                                        \
        out[i+3] = func(in0[i+3], in1[i+3]);                                        \
        out[i+4] = func(in0[i+4], in1[i+4]);                                        \
        out[i+5] = func(in0[i+5], in1[i+5]);                                        \
        out[i+6] = func(in0[i+6], in1[i+6]);                                        \
        out[i+7] = func(in0[i+7], in1[i+7]);                                        \
    }                                                                               \
    for (; i < n; i++) {                                                            \
        out[i] = func(in0[i], in1[i]);                                              \
    }                                                                               \
}

#define GPU_UNARY(func, t0, t1)                                 \
void gpu_##func##_##t0##_##t1(int n, t0##_t* in0, t1##_t* out)  \
{                                                               \
    for (int i = 0; i < n; i ++)                                \
        out[i] = func(in0[i]);                                  \
}

#else

#define GPU_BINARY(func, t0, t1, t2)                                                \
__global__                                                                          \
void _##func##_##t0##_##t1##_##t2(int n, t0##_t* in0, t1##_t* in1, t2##_t* out)     \
{                                                                                   \
    int index = blockIdx.x * blockDim.x + threadIdx.x;                              \
    int stride = blockDim.x * gridDim.x;                                            \
    for (int i = index; i < n; i += stride)                                         \
        out[i] = func(in0[i], in1[i]);                                              \
}                                                                                   \
void gpu_##func##_##t0##_##t1##_##t2(int n, t0##_t* in0, t1##_t* in1, t2##_t* out)  \
{                                                                                   \
    int blockSize = 256;                                                            \
    int numBlocks = (n + blockSize - 1) / blockSize;                                \
    _##func##_##t0##_##t1##_##t2<<<numBlocks, blockSize>>>(n, in0, in1, out);       \
}

#define GPU_UNARY(func, t0, t1)                                     \
__global__                                                          \
void _##func##_##t0##_##t1(int n, t0##_t* in0, t1##_t* out)         \
{                                                                   \
    int index = blockIdx.x * blockDim.x + threadIdx.x;              \
    int stride = blockDim.x * gridDim.x;                            \
    for (int i = index; i < n; i += stride)                         \
        out[i] = func(in0[i]);                                      \
}                                                                   \
void gpu_##func##_##t0##_##t1(int n, t0##_t* in0, t1##_t* out)      \
{                                                                   \
    int blockSize = 256;                                            \
    int numBlocks = (n + blockSize - 1) / blockSize;                \
    _##func##_##t0##_##t1<<<numBlocks, blockSize>>>(n, in0, out);   \
}

#endif

#define GPU_ALL_BINARY(name)                    \
    GPU_BINARY(name, int8, int8, int8)          \
    GPU_BINARY(name, int8, int16, int16)        \
    GPU_BINARY(name, int8, int32, int32)        \
    GPU_BINARY(name, int8, int64, int64)        \
    GPU_BINARY(name, int8, uint8, int16)        \
    GPU_BINARY(name, int8, uint16, int32)       \
    GPU_BINARY(name, int8, uint32, int64)       \
    GPU_BINARY(name, int8, float32, float32)    \
    GPU_BINARY(name, int8, float64, float64)    \
                                                \
    GPU_BINARY(name, int16, int8, int16)        \
    GPU_BINARY(name, int16, int16, int16)       \
    GPU_BINARY(name, int16, int32, int32)       \
    GPU_BINARY(name, int16, int64, int64)       \
    GPU_BINARY(name, int16, uint8, int16)       \
    GPU_BINARY(name, int16, uint16, int32)      \
    GPU_BINARY(name, int16, uint32, int64)      \
    GPU_BINARY(name, int16, float32, float32)   \
    GPU_BINARY(name, int16, float64, float64)   \
                                                \
    GPU_BINARY(name, int32, int8, int32)        \
    GPU_BINARY(name, int32, int16, int32)       \
    GPU_BINARY(name, int32, int32, int32)       \
    GPU_BINARY(name, int32, int64, int64)       \
    GPU_BINARY(name, int32, uint8, int32)       \
    GPU_BINARY(name, int32, uint16, int32)      \
    GPU_BINARY(name, int32, uint32, int64)      \
    GPU_BINARY(name, int32, float64, float64)   \
                                                \
    GPU_BINARY(name, int64, int8, int64)        \
    GPU_BINARY(name, int64, int16, int64)       \
    GPU_BINARY(name, int64, int32, int64)       \
    GPU_BINARY(name, int64, int64, int64)       \
    GPU_BINARY(name, int64, uint8, int64)       \
    GPU_BINARY(name, int64, uint16, int64)      \
    GPU_BINARY(name, int64, uint32, int64)      \
                                                \
    GPU_BINARY(name, uint8, int8, int16)        \
    GPU_BINARY(name, uint8, int16, int16)       \
    GPU_BINARY(name, uint8, int32, int32)       \
    GPU_BINARY(name, uint8, int64, int64)       \
    GPU_BINARY(name, uint8, uint8, uint8)       \
    GPU_BINARY(name, uint8, uint16, uint16)     \
    GPU_BINARY(name, uint8, uint32, uint32)     \
    GPU_BINARY(name, uint8, uint64, uint64)     \
    GPU_BINARY(name, uint8, float32, float32)   \
    GPU_BINARY(name, uint8, float64, float64)   \
                                                \
    GPU_BINARY(name, uint16, int8, int32)       \
    GPU_BINARY(name, uint16, int16, int32)      \
    GPU_BINARY(name, uint16, int32, int32)      \
    GPU_BINARY(name, uint16, int64, int64)      \
    GPU_BINARY(name, uint16, uint8, uint16)     \
    GPU_BINARY(name, uint16, uint16, uint16)    \
    GPU_BINARY(name, uint16, uint32, uint32)    \
    GPU_BINARY(name, uint16, uint64, uint64)    \
    GPU_BINARY(name, uint16, float32, float32)  \
    GPU_BINARY(name, uint16, float64, float64)  \
                                                \
    GPU_BINARY(name, uint32, int8, int64)       \
    GPU_BINARY(name, uint32, int16, int64)      \
    GPU_BINARY(name, uint32, int32, int64)      \
    GPU_BINARY(name, uint32, int64, int64)      \
    GPU_BINARY(name, uint32, uint8, uint32)     \
    GPU_BINARY(name, uint32, uint16, uint16)    \
    GPU_BINARY(name, uint32, uint32, uint32)    \
    GPU_BINARY(name, uint32, uint64, uint64)    \
    GPU_BINARY(name, uint32, float64, float64)  \
                                                \
    GPU_BINARY(name, uint64, uint8, uint64)     \
    GPU_BINARY(name, uint64, uint16, uint64)    \
    GPU_BINARY(name, uint64, uint32, uint64)    \
    GPU_BINARY(name, uint64, uint64, uint64)    \
                                                \
    GPU_BINARY(name, float32, int8, float32)    \
    GPU_BINARY(name, float32, int16, float32)   \
    GPU_BINARY(name, float32, uint8, float32)   \
    GPU_BINARY(name, float32, uint16, float32)  \
    GPU_BINARY(name, float32, float32, float32) \
    GPU_BINARY(name, float32, float64, float64) \
                                                \
    GPU_BINARY(name, float64, int8, float64)    \
    GPU_BINARY(name, float64, int16, float64)   \
    GPU_BINARY(name, float64, int32, float64)   \
    GPU_BINARY(name, float64, uint8, float64)   \
    GPU_BINARY(name, float64, uint16, float64)  \
    GPU_BINARY(name, float64, uint32, float64)  \
    GPU_BINARY(name, float64, float32, float64) \
    GPU_BINARY(name, float64, float64, float64)

#define add(x, y) x + y
#define subtract(x, y) x - y
#define multiply(x, y) x * y
#define divide(x, y) x / y

#define GPU_ALL_UNARY_FLOAT(name)        \
    GPU_UNARY(name##f, int8, float32)    \
    GPU_UNARY(name##f, int16, float32)   \
    GPU_UNARY(name##f, uint8, float32)   \
    GPU_UNARY(name##f, uint16, float32)  \
    GPU_UNARY(name##f, float32, float32) \
/*  GPU_UNARY(name, int32, float64)      \
    GPU_UNARY(name, uint32, float64)     \
*/  GPU_UNARY(name, float64, float64)

#endif
