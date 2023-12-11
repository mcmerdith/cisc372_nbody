#include "cuda_compute.hpp"
#include "config.hpp"

/* local computation variables */

vector3 *accelerations, *accel_sum;
double magnitude_sq;

/* device copies of input/output variables */

extern vector3 *hVel, *hPos;
vector3 *d_hVel, *d_hPos;

extern double *mass;
double *d_mass;

/* KERNEL DIMENSIONS */

// dim3 computeBlockSize(32, 32);
dim3 computeBlockSize(18, 18, 3);
dim3 computeBlockCount(CEIL_DIVIDE(NUMENTITIES, DIM_SIZE(computeBlockSize)), CEIL_DIVIDE(NUMENTITIES, DIM_SIZE(computeBlockSize)));

// dim3 sumBlockSize(341, 3);
dim3 sumBlockSize(18, 18, 3);
dim3 sumBlockCount(CEIL_DIVIDE(NUMENTITIES, DIM_SIZE(sumBlockSize)), CEIL_DIVIDE(NUMENTITIES, DIM_SIZE(sumBlockSize)));

dim3 updateBlockSize(341, 1, 3);
dim3 updateBlockCount(CEIL_DIVIDE(NUMENTITIES, DIM_SIZE(updateBlockSize)));

/* KERNELS */

__global__ void compute_accelerations()
{
    // make an acceleration matrix which is NUMENTITIES squared in size;
    int i = blockIdx.x * blockDim.x + threadIdx.x,
        j = blockIdx.y * blockDim.y + threadIdx.y,
        k = threadIdx.z;
    // only work inside the grid
    if (i >= NUMENTITIES || j >= NUMENTITIES)
        return;

    // first compute the pairwise accelerations.  Effect is on the first argument.
    if (i == j)
    {
        accelerations[INDEX(i, j)][k] = 0;
    }
    else
    {
        // vector3 distance;
        double distance = hPos[i][k] - hPos[j][k];

        // calculate magnitude components
        __shared__ double magnitude_sq[blockDim.x][blockDim.y];
        double *l_mag_sq = &magnitude_sq[threadIdx.x][threadIdx.y];
        atomicAdd(l_mag_sq, SQUARE(distance));

        // All threads must compute their component of magnitude sq before continuing
        __syncthreads();

        // calculate magnitudes
        double magnitude = sqrt(*l_mag_sq);
        double accelmag = -1 * GRAV_CONSTANT * mass[j] / (*l_mag_sq);

        accelerations[INDEX(i, j)][k] = accelmag * distance / magnitude;
    }
}

__global__ void sum_matrix()
{
    // sum up the rows of our matrix to get effect on each entity, then update velocity and position.
    int i = blockIdx.x * blockDim.x + threadIdx.x,
        j = blockIdx.y * blockDim.y + threadIdx.y,
        k = threadIdx.z;

    if (i >= NUMENTITIES || j >= NUMENTITIES)
        return;

    // for (j = 0; j < NUMENTITIES; j++)
    //     for (k = 0; k < 3; k++)
    // accel_sum[i][k] += accelerations[INDEX(i, j)][k];
    atomicAdd(&accel_sum[i][k], accelerations[INDEX(i, j)][k]);
}

__global__ void update_positions()
{
    // compute the new velocity based on the acceleration and time interval
    // compute the new position based on the velocity and time interval
    int i = blockIdx.x * blockDim.x + threadIdx.x,
        k = threadIdx.z;

    if (i >= NUMENTITIES)
        return;

    // for (k = 0; k < 3; k++)
    // {
    hVel[i][k] = accel_sum[i][k] * INTERVAL;
    hPos[i][k] += hVel[i][k] * INTERVAL;
    // }
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
    // allocates shared memory for a matrix of magnitudes for each pair of entities calculated
    compute_accelerations<<<computeBlockCount, computeBlockSize, sizeof(double) * computeBlockSize.x * computeBlockSize.y>>>();
    // TODO use reduction within shared memory to improve performance
    sum_matrix<<<sumBlockCount, sumBlockSize>>>();
    update_positions<<<updateBlockCount, updateBlockSize>>>();
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