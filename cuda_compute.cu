#include "cuda_compute.hpp"
#include "config.hpp"
#include <stdio.h>

#include <iostream>
using namespace std;

/* local computation variables */

vector3 *accel_values, **accels, *accel_sum;

/* device copies of input/output variables */

vector3 *d_hVel, *d_hPos;
double *d_mass;

/* KERNEL DIMENSIONS */

dim3 computeBlockSize(32, 32);
dim3 computeBlockCount(CEIL_DIVIDE(NUMENTITIES, computeBlockSize.x), CEIL_DIVIDE(NUMENTITIES, computeBlockSize.y));

dim3 sumBlockSize(1, 341, 3);
dim3 sumBlockCount(1, CEIL_DIVIDE(NUMENTITIES, sumBlockSize.y));

dim3 updateBlockSize(341, 1, 3);
dim3 updateBlockCount(CEIL_DIVIDE(NUMENTITIES, updateBlockSize.x));

/* KERNELS */

__global__ void compute_accelerations(vector3 **d_accels, vector3 *d_hPos, double *d_mass)
{
    // make an acceleration matrix which is NUMENTITIES squared in size;
    int i = blockIdx.y * blockDim.y + threadIdx.y,
        j = blockIdx.x * blockDim.x + threadIdx.x,
        k;

    // only work inside the grid
    if (i >= NUMENTITIES || j >= NUMENTITIES)
        return;

    // first compute the pairwise accelerations.  Effect is on the first argument.
    if (i == j)
    {
        for (k = 0; k < 3; k++)
            d_accels[i][j][k] = 0;
    }
    else
    {
        vector3 distance;
        for (k = 0; k < 3; k++)
            distance[k] = d_hPos[i][k] - d_hPos[j][k];

        double magnitude_sq = 0;

        for (k = 0; k < 3; k++)
            magnitude_sq += SQUARE(distance[k]);

        // calculate magnaitudes
        double magnitude = sqrt(magnitude_sq);
        double accelmag = -1 * GRAV_CONSTANT * d_mass[j] / (magnitude_sq);

        for (k = 0; k < 3; k++)
            d_accels[i][j][k] = accelmag * distance[k] / magnitude;
    }
}

__global__ void sum_matrix(vector3 *d_accel_sum, vector3 **d_accels)
{
    // sum up the rows of our matrix to get effect on each entity, then update velocity and position.
    int i = blockIdx.y * blockDim.y + threadIdx.y,
        k = threadIdx.z;

    if (i >= NUMENTITIES)
        return;

    for (int j = 0; j < NUMENTITIES; j++)
        d_accel_sum[i][k] += d_accels[i][j][k];
}

__global__ void update_positions(vector3 *d_hVel, vector3 *d_hPos, vector3 *d_accel_sum)
{
    // compute the new velocity based on the acceleration and time interval
    // compute the new position based on the velocity and time interval
    int i = blockIdx.x * blockDim.x + threadIdx.x,
        k = threadIdx.z;

    if (i >= NUMENTITIES)
        return;

    d_hVel[i][k] += d_accel_sum[i][k] * INTERVAL;
    d_hPos[i][k] += d_hVel[i][k] * INTERVAL;
}

__global__ void map_accel_values(vector3 *accel_values, vector3 **accels)
{
    for (int i = 0; i < NUMENTITIES; i++)
        accels[i] = &accel_values[i * NUMENTITIES];
}

/* PUBLIC FUNCTIONS */

void compute_prepare()
{
    // allocate the buffers to hold computation data between runs
    cudaMalloc(&accel_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
    cudaMalloc(&accels, sizeof(vector3 *) * NUMENTITIES);
    cudaMalloc(&accel_sum, sizeof(vector3) * NUMENTITIES);

    map_accel_values<<<1, 1>>>(accel_values, accels);
    cudaDeviceSynchronize();

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
    compute_accelerations<<<computeBlockCount, computeBlockSize>>>(accels, d_hPos, d_mass);
    sum_matrix<<<sumBlockCount, sumBlockSize>>>(accel_sum, accels);
    update_positions<<<updateBlockCount, updateBlockSize>>>(d_hVel, d_hPos, accel_sum);
}

void compute_complete()
{
    // copy the device data back to the host
    cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);

    // free the computation data buffers
    cudaFree(accel_values);
    cudaFree(accels);
    cudaFree(accel_sum);

    // free the device copies of inputs and outputs
    cudaFree(d_hVel);
    cudaFree(d_hPos);
    cudaFree(d_mass);
}