#ifndef gpu_mem_h
#define gpu_mem_h

#include <stdint.h>

#ifdef FAKE_GPU
#include <string.h>
#include <stdlib.h>
#endif

typedef float float32_t;
typedef double float64_t;

#ifdef FAKE_GPU

#define NEW_ARRAY(t)                            \
t##_t* new_array_##t(int n)                     \
{                                               \
    t##_t* x;                                   \
    x = (t##_t*)malloc(n * sizeof(t##_t));      \
    return x;                                   \
}

#define DEL_ARRAY(t)                    \
void del_array_##t(t##_t* x)            \
{                                       \
    free(x);                            \
}

#define COPY_ARRAY(t)                                           \
void copy_array_##t(t##_t* dst,t##_t* src, int n)               \
{                                                               \
    memcpy(dst, src, n * sizeof(t##_t));                        \
}

void synchro();

#else

#define NEW_ARRAY(t)                            \
extern "C" t##_t* new_array_##t(int n)          \
{                                               \
    t##_t* x;                                   \
    cudaMallocManaged(&x, n * sizeof(t##_t));   \
    return x;                                   \
}

#define DEL_ARRAY(t)                    \
extern "C" void del_array_##t(t##_t* x) \
{                                       \
    cudaFree(x);                        \
}

#define COPY_ARRAY(t)                                           \
extern "C" void copy_array_##t(t##_t* dst,t##_t* src, int n)    \
{                                                               \
    memcpy(dst, src, n * sizeof(t##_t));                        \
}

extern "C" void synchro();

#endif

#define ALL_NEW_ARRAY   \
    NEW_ARRAY(int8)     \
    NEW_ARRAY(int16)    \
    NEW_ARRAY(int32)    \
    NEW_ARRAY(int64)    \
    NEW_ARRAY(uint8)    \
    NEW_ARRAY(uint16)   \
    NEW_ARRAY(uint32)   \
    NEW_ARRAY(uint64)   \
    NEW_ARRAY(float32)  \
    NEW_ARRAY(float64)

#define ALL_DEL_ARRAY   \
    DEL_ARRAY(int8)     \
    DEL_ARRAY(int16)    \
    DEL_ARRAY(int32)    \
    DEL_ARRAY(int64)    \
    DEL_ARRAY(uint8)    \
    DEL_ARRAY(uint16)   \
    DEL_ARRAY(uint32)   \
    DEL_ARRAY(uint64)   \
    DEL_ARRAY(float32)  \
    DEL_ARRAY(float64)

#define ALL_COPY_ARRAY  \
    COPY_ARRAY(int8)    \
    COPY_ARRAY(int16)   \
    COPY_ARRAY(int32)   \
    COPY_ARRAY(int64)   \
    COPY_ARRAY(uint8)   \
    COPY_ARRAY(uint16)  \
    COPY_ARRAY(uint32)  \
    COPY_ARRAY(uint64)  \
    COPY_ARRAY(float32) \
    COPY_ARRAY(float64)

#endif
