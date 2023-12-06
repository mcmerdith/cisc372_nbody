#pragma once
#include "vector.hpp"

#define FILL_CUDA_VEC(vector, xv, yv, zv) \
    {                                     \
        vector.x = xv;                    \
        vector.y = yv;                    \
        vector.z = zv;                    \
    }

#define CUDA_VEC_DIFFERENCE(result, vec1, vec2) \
    {                                           \
        result.x = vec1.x - vec2.x;             \
        result.y = vec1.y - vec2.y;             \
        result.z = vec1.z - vec2.z;             \
    }

#define SQUARE(x) (x * x)

#define INDEX(i, j) i *NUMENTITIES + j

__global__ void compute_accelerations(vector3 *accelerations, vector3 *hPos, double *mass);
__global__ void sum_matrix(vector3 *accel_sum, vector3 *accelerations);
__global__ void update_positions(vector3 *accel_sum, vector3 *hVel, vector3 *hPos);