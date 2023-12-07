#include "cuda_compute.hpp"
#include "config.hpp"

__global__ void compute_accelerations(vector3 *accelerations, vector3 *hPos, double *mass)
{
    // make an acceleration matrix which is NUMENTITIES squared in size;
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, k;
    // only work inside the grid
    if (i >= NUMENTITIES || j >= NUMENTITIES)
        return;

    // first compute the pairwise accelerations.  Effect is on the first argument.
    if (i == j)
    {
        FILL_VECTOR(accelerations[INDEX(i, j)], 0, 0, 0);
    }
    else
    {
        vector3 distance;
        double magnitude_sq = 0;
        for (k = 0; k < 3; k++)
        {
            distance[k] = hPos[i][k] - hPos[j][k];
            magnitude_sq += SQUARE(distance[k]);
        }
        double magnitude = sqrt(magnitude_sq);
        double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
        FILL_VECTOR(accelerations[INDEX(i, j)],
                    accelmag * distance[0] / magnitude,
                    accelmag * distance[1] / magnitude,
                    accelmag * distance[2] / magnitude);
    }
}

__global__ void sum_matrix(vector3 *accel_sum, vector3 *accelerations)
{
    // sum up the rows of our matrix to get effect on each entity, then update velocity and position.
    int i = blockIdx.x * blockDim.x + threadIdx.x, k;
    if (i >= NUMENTITIES)
        return;

    for (int j = 0; j < NUMENTITIES; j++)
        for (int k = 0; k < 3; k++)
            accel_sum[i][k] += accelerations[INDEX(i, j)][k];
}

__global__ void update_positions(vector3 *accel_sum, vector3 *hVel, vector3 *hPos)
{
    // compute the new velocity based on the acceleration and time interval
    // compute the new position based on the velocity and time interval
    int i = blockIdx.x * blockDim.x + threadIdx.x, k;
    if (i >= NUMENTITIES)
        return;

    for (k = 0; k < 3; k++)
    {
        hVel[i][k] = accel_sum[i][k] * INTERVAL;
        hPos[i][k] += hVel[i][k] * INTERVAL;
    }
}