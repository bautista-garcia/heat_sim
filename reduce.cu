#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "cuda_utils.h"

// Global variables
float *d_data, *h_data;
int N;

// TODO: Implement your reduction kernel here
// This is a placeholder - replace with your actual reduction implementation
__global__ void reduction_kernel(const float* __restrict__ input, float* __restrict__ output, int n) {
    extern __shared__ float sdata[];
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
    sdata[tid] = input[i] + input[i+blockDim.x];
    __syncthreads();
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

double dwalltime(){
    double sec;
    struct timeval tv;

    gettimeofday(&tv,NULL);
    sec = tv.tv_sec + tv.tv_usec/1000000.0;
    return sec;
}

void initialize_grid(int grid_size) {
    N = grid_size;
    int total_elements = N * N;
    
    // Allocate host memory
    h_data = (float*)malloc(sizeof(float) * total_elements);
    
    // Initialize with some values (e.g., sequential or random)
    for (int i = 0; i < total_elements; i++) {
        h_data[i] = 1.0f; // Or use: (float)(i % 100) for varied values
    }
    
    // Allocate device memory
    size_t bytes = sizeof(float) * total_elements;
    cudaMalloc(&d_data, bytes);
    
    // Copy data to device
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    
    printf("Initialized %dx%d grid (%d elements)\n", N, N, total_elements);
}

int main(int argc, char** argv) {
    // Device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    PRINT_DEVICE_SUMMARY(prop);

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <grid_size> [threads_per_block]\n", argv[0]);
        return 1;
    }

    // Parameters
    int grid_size = atoi(argv[1]);
    int total_elements = grid_size * grid_size;
    int threads_per_block = (argc >= 3) ? atoi(argv[2]) : 128;
    
    if (grid_size <= 0) {
        fprintf(stderr, "Grid size must be positive.\n");
        return 1;
    }

    int num_blocks = CEIL_DIV(total_elements, threads_per_block * 2); // 2 because of the first reduction step
    int shm_bytes = threads_per_block * sizeof(float);

    // Initialize grid
    initialize_grid(grid_size);
    // Allocate device memory for result (one entry per block)
    float *d_result;
    cudaMalloc(&d_result, sizeof(float) * num_blocks);
    cudaMemset(d_result, 0, sizeof(float) * num_blocks);

    printf("\nKernel configuration:\n");
    printf("Total elements: %d\n", total_elements);
    printf("Threads per block: %d\n", threads_per_block);
    printf("Number of blocks: %d\n", num_blocks);
    printf("Shared memory per block: %d bytes\n", shm_bytes);

    // Measure kernel execution time
    double ti = dwalltime();
    reduction_kernel<<<num_blocks, threads_per_block, shm_bytes>>>(d_data, d_result, total_elements);
    cudaDeviceSynchronize();
    double tf = dwalltime();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        cudaFree(d_result);
        free(h_data);
        return 1;
    }
    
    // Copy partial results back
    float *h_partials = (float*)malloc(sizeof(float) * num_blocks);
    cudaMemcpy(h_partials, d_result, sizeof(float) * num_blocks, cudaMemcpyDeviceToHost);

    // Reduce partials on host
    float result = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        result += h_partials[i];
    }
    
    printf("\n======================= RESULTS =======================\n");
    printf("Kernel execution time: %.6f seconds\n", tf - ti);
    printf("Reduction result: %.6f\n", result);
    printf("Expected sum (if all 1.0): %.6f\n", (float)total_elements);
    printf("========================================================\n");
    
    // Cleanup
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
    free(h_partials);
    
    return 0;
}

