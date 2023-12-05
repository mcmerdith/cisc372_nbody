#pragma once

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

__global__ void compute_accelerations(double3 **accelerations, double3 *hPos, double *mass);
__global__ void sum_matrix(double3 *accel_sum, double3 **accelerations);
__global__ void update_positions(double3 *accel_sum, double3 *hVel, double3 *hPos);