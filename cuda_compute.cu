#include "cuda_compute.hpp"
#include "config.hpp"

vector3 *accelerations, *accel_sum;

extern vector3 *hVel, *hPos;
vector3 *d_hVel, *d_hPos;

extern double *mass;
double *d_mass;

dim3 computeBlockSize(32, 32);
dim3 computeBlockCount((NUMENTITIES + computeBlockSize.x - 1) / computeBlockSize.x, (NUMENTITIES + computeBlockSize.y - 1) / computeBlockSize.y);

dim3 sumBlockSize(1024);
dim3 sumBlockCount((NUMENTITIES + sumBlockSize.x - 1) / sumBlockSize.x);

dim3 updateBlockSize(1024);
dim3 updateBlockCount((NUMENTITIES + updateBlockSize.x - 1) / updateBlockSize.x);

/* KERNELS */

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
    int i = blockIdx.x * blockDim.x + threadIdx.x, j, k;
    if (i >= NUMENTITIES)
        return;

    for (j = 0; j < NUMENTITIES; j++)
        for (k = 0; k < 3; k++)
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

/* PUBLIC FUNCTIONS */

void compute_prepare()
{
    // allocate the buffers to hold computation data between runs
    cudaMalloc(&accelerations, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
    cudaMalloc(&accel_sum, sizeof(vector3) * NUMENTITIES);

    // allocate device copies of the inputs and outputs
    cudaMalloc(&d_hVel, sizeof(vector3) * NUMENTITIES);
    cudaMalloc(&d_hPos, sizeof(vector3) * NUMENTITIES);
    cudaMalloc(&d_mass, sizeof(double) * NUMENTITIES);

    // copy the host data over to the device
    cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);
}

void compute()
{
    compute_accelerations<<<computeBlockCount, computeBlockSize>>>(accelerations, d_hPos, d_mass);
    sum_matrix<<<sumBlockCount, sumBlockSize>>>(accel_sum, accelerations);
    update_positions<<<updateBlockCount, updateBlockSize>>>(accel_sum, d_hVel, d_hPos);
}

void compute_complete()
{
    // copy the device data back to the host
    cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);

    // free the computation data buffers
    cudaFree(accelerations);
    cudaFree(accel_sum);

    // free the device copies of inputs and outputs
    cudaFree(d_hVel);
    cudaFree(d_hPos);
    cudaFree(d_mass);
}