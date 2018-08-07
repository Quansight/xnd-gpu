===========
xnd-gpu API
===========

-------------------------------------
Creating an xnd container for the GPU
-------------------------------------

For an ``xnd`` container data to be accessible by the GPU, it has to be created
using the ``xnd_gpu`` function, which allocates memory in the `unified memory
<https://devblogs.nvidia.com/unified-memory-in-cuda-6>`_. This memory provides a
single view of the physically separate CPU and GPU memories, and a mechanism
that automatically handles copies between the two. For instance, when the CPU
writes data to the unified memory, the GPU will transparently see this data when
it reads the content of the unified memory. Under the hood, the data has been
copied on demand from the CPU memory to the GPU memory. "On demand" means that
when the GPU tried to read, the system detected that its data (in its own
memory) was not up to date, and triggered a memory copy. This copy was not done
until the GPU needed to access the data. Furthermore, only the relevant part of
the data was copied (the corresponding page). The same mechanism works in the
other direction, when the GPU writes data to the unified memory and the CPU
reads it. This mechanism makes sure the CPU and GPU memories are kept in sync.

``xnd_gpu`` can be called with different arguments, depending on whether you
want to get an empty container or you want it to be initialized.

.. code-block:: python

   from xnd_gpu import xnd_gpu

   # get an empty xnd container from shape and data type
   x0 = xnd_gpu(shape=(2, 3), dtype='float64')
   # xnd([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], type='2 * 3 * float64')

   # get an empty xnd container from a NumPy array's shape and data type
   a = np.empty((2, 3), dtype='float64')
   x0 = xnd_gpu(empty_like=a)
   # xnd([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], type='2 * 3 * float64')

   # get an xnd container initialized from a NumPy array
   a = np.arange(6, dtype='float64').reshape(2, 3)
   x0 = xnd_gpu(from_buffer=a)
   # xnd([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], type='2 * 3 * float64')

--------------------------
Calling GPU gumath kernels
--------------------------

Once you have GPU-compatible ``xnd`` containers, you can call ``gumath`` kernels
on them. Today the only way to get results back is to write them in-place in a
GPU ``xnd`` container that is passed to the function, a lot like NumPy's ``out``
parameter (here it is just passed as the last argument of the function). All of
``gumath``'s builtin kernels are available on the GPU.

Note that a major difference with ``gumath`` kernels which run on the CPU is
that GPU ``gumath`` kernels are asynchronous: a function call will return
immediately. This allows queueing functions and not having to wait for them to
finish before doing something else. In particular, one can use the CPU in
parallel, which leads to a better utilization of ressources and better overall
performances. But if the results of the GPU computations are needed on the CPU
side, one must wait for the GPU to finish. This is achieved by an explicit call
to the ``gpu_synchro`` function, which blocks until the GPU is done with all
the computations. Note that omitting to call this function prior to accessing
data on a GPU ``xnd`` container will lead to potentially corrupted data.

.. code-block:: python

   import numpy as np
   from xnd_gpu import xnd_gpu
   from gpu_func import gpu_add, gpu_synchro

   a = np.arange(6, dtype='float64').reshape(2, 3)
   in0 = xnd_gpu(from_buffer=a)
   in1 = xnd_gpu(from_buffer=a)
   out = xnd_gpu(empty_like=a) # placeholder for the result
   print(in0)
   # xnd([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], type='2 * 3 * float64')
   gpu_add(in0, in1, out) # asynchronous GPU kernel call
   # GPU and CPU execute in parallel
   gpu_synchro() # explicitly block until GPU is done
   # it is now safe to access the GPU result
   print(out)
   # xnd([[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]], type='2 * 3 * float64')
