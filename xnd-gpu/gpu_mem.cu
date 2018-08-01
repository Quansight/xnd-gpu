#include "gpu_mem.h"

ALL_NEW_ARRAY
ALL_DEL_ARRAY
ALL_COPY_ARRAY

void synchro()
{
#ifndef FAKE_GPU
    cudaDeviceSynchronize();
#endif
}
