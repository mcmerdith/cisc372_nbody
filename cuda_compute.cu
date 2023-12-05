#include "cuda_compute.hpp"
#include "config.hpp"

__global__ void compute_accelerations(double3 **accelerations, double3 *hPos, double *mass)
{
    // make an acceleration matrix which is NUMENTITIES squared in size;
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    // only work inside the grid
    if (i >= NUMENTITIES || j >= NUMENTITIES)
        return;
    // first compute the pairwise accelerations.  Effect is on the first argument.
    if (i == j)
    {
        FILL_CUDA_VEC(accelerations[i][j], 0, 0, 0);
    }
    else
    {
        double3 distance;
        CUDA_VEC_DIFFERENCE(distance, hPos[i], hPos[j]);
        double magnitude_sq = SQUARE(distance.x) + SQUARE(distance.y) + SQUARE(distance.z);
        double magnitude = sqrt(magnitude_sq);
        double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
        FILL_CUDA_VEC(accelerations[i][j],
                      accelmag * distance.x / magnitude,
                      accelmag * distance.y / magnitude,
                      accelmag * distance.z / magnitude);
    }
}

__global__ void sum_matrix(double3 *accel_sum, double3 **accelerations)
{
    // sum up the rows of our matrix to get effect on each entity, then update velocity and position.
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    accel_sum[i].x += accelerations[i][j].x;
    accel_sum[i].y += accelerations[i][j].y;
    accel_sum[i].z += accelerations[i][j].z;
}

__global__ void update_positions(double3 *accel_sum, double3 *hVel, double3 *hPos)
{
    // compute the new velocity based on the acceleration and time interval
    // compute the new position based on the velocity and time interval
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    hVel[i].x = accel_sum[i].x * INTERVAL;
    hPos[i].x += hVel[i].x * INTERVAL;
    hVel[i].y = accel_sum[i].y * INTERVAL;
    hPos[i].y += hVel[i].y * INTERVAL;
    hVel[i].z = accel_sum[i].z * INTERVAL;
    hPos[i].z += hVel[i].z * INTERVAL;
}