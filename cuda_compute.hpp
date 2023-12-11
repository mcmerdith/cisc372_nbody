#pragma once
#include "compute.hpp"
#include "vector.hpp"

#define DIM_SIZE(dim) (dim.x * dim.y * dim.z)

#define CEIL_DIVIDE(x, y) ((x + y - 1) / y)

#define SQUARE(x) (x * x)

#define INDEX(i, j) ((i * NUMENTITIES) + j)

// #if __CUDA_ARCH__ < 600
// /**
//  * https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicadd
//  * Code taken from NVIDIA CUDA Docs for ARCH < 600
//  */
// __device__ double atomicAdd(double *address, double val)
// {
//     unsigned long long int *address_as_ull =
//         (unsigned long long int *)address;
//     unsigned long long int old = *address_as_ull, assumed;

//     do
//     {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed,
//                         __double_as_longlong(val +
//                                              __longlong_as_double(assumed)));

//         // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//     } while (assumed != old);

//     return __longlong_as_double(old);
// }
// #endif

extern __device__ double atomicAdd(double *address, double val);

void compute_prepare();
void compute_complete();