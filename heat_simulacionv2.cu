#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "heat_simulation.h"

#define PADDING_GLOBAL 2
#define TILE_X 32
#define THREADS_PER_BLOCK 256
#define REDUCTION_STOP_THRESHOLD 0.0001f

float* new_grid;

float diffusion_rate = 0.25f;

float *grid = NULL;
int grid_size = 0;

// GPU buffers
static float* deviceGridCurrent = NULL;
static float* deviceGridNext = NULL;
static float* deviceFlatBuffer = NULL;  // For reduction
static float* deviceReductionPartials = NULL;
static float* hostReductionPartials = NULL;
static int pitchElements = 0;
static int reductionBlocks = 0;
static float previousReduction = 0.0f;
static float currentReduction = 0.0f;
static bool initialized = false;

// Helper function to copy from pitched device buffer to flat host buffer
static void copyPitchedToFlat(float* pitchedDevice, float* flatHost, int gridSize, int pitch) {
    float* tempHost = (float*)malloc(sizeof(float) * pitch * pitch);
    cudaMemcpy(tempHost, pitchedDevice, sizeof(float) * pitch * pitch, cudaMemcpyDeviceToHost);
    
    int baseOffset = pitch + 1;
    for (int y = 0; y < gridSize; y++) {
        for (int x = 0; x < gridSize; x++) {
            flatHost[y * gridSize + x] = tempHost[baseOffset + y * pitch + x];
        }
    }
    free(tempHost);
}

// Helper function to copy from flat host buffer to pitched device buffer
static void copyFlatToPitched(float* flatHost, float* pitchedDevice, int gridSize, int pitch) {
    float* tempHost = (float*)calloc(pitch * pitch, sizeof(float));
    
    int baseOffset = pitch + 1;
    for (int y = 0; y < gridSize; y++) {
        for (int x = 0; x < gridSize; x++) {
            tempHost[baseOffset + y * pitch + x] = flatHost[y * gridSize + x];
        }
    }
    
    cudaMemcpy(pitchedDevice, tempHost, sizeof(float) * pitch * pitch, cudaMemcpyHostToDevice);
    free(tempHost);
}

// ============================================================================
// KERNELS
// ============================================================================

__global__ void applyHeatSources(float* grid, int gridSize, int pitchElements) {
    int centerX = gridSize / 2;
    int centerY = gridSize / 2;
    int baseOffset = pitchElements + 1;

    grid[baseOffset + centerY * pitchElements + centerX] = 100.0f;

    int sourceOffset = 20;
    if (centerY + sourceOffset < gridSize && centerX + sourceOffset < gridSize) {
        grid[baseOffset + (centerY + sourceOffset) * pitchElements + (centerX + sourceOffset)] = 100.0f;
    }
    if (centerY + sourceOffset < gridSize && centerX >= sourceOffset) {
        grid[baseOffset + (centerY + sourceOffset) * pitchElements + (centerX - sourceOffset)] = 100.0f;
    }
    if (centerY >= sourceOffset && centerX + sourceOffset < gridSize) {
        grid[baseOffset + (centerY - sourceOffset) * pitchElements + (centerX + sourceOffset)] = 100.0f;
    }
    if (centerY >= sourceOffset && centerX >= sourceOffset) {
        grid[baseOffset + (centerY - sourceOffset) * pitchElements + (centerX - sourceOffset)] = 100.0f;
    }
}

__global__ void simulateDiffusion(const float* __restrict__ gridCurrent,
                                  float* __restrict__ gridNext,
                                  int gridSize,
                                  int pitchElements,
                                  float diffusionRate) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= gridSize || y >= gridSize) {
        return;
    }

    const float* current = gridCurrent + pitchElements + 1;
    float* output = gridNext + pitchElements + 1;

    extern __shared__ float sharedMemory[];
    const int strideY = blockDim.x + 3;
    const int localX = threadIdx.x + 1;
    const int localY = threadIdx.y + 1;
    int index = y * pitchElements + x;

    sharedMemory[localY * strideY + localX] = current[index];
    if (threadIdx.y == 0) sharedMemory[localX] = current[index - pitchElements];
    if (threadIdx.y == blockDim.y - 1) sharedMemory[(blockDim.y + 1) * strideY + localX] = current[index + pitchElements];
    if (threadIdx.x == 0) sharedMemory[localY * strideY] = current[index - 1];
    if (threadIdx.x == blockDim.x - 1) sharedMemory[localY * strideY + (blockDim.x + 1)] = current[index + 1];

    __syncthreads();

    float center = sharedMemory[localY * strideY + localX];
    float up = sharedMemory[(localY - 1) * strideY + localX];
    float down = sharedMemory[(localY + 1) * strideY + localX];
    float left = sharedMemory[localY * strideY + (localX - 1)];
    float right = sharedMemory[localY * strideY + (localX + 1)];

    output[index] = center + diffusionRate * (up + down + left + right - 4.0f * center);
}

__global__ void reduceGridKernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int elementCount) {
    extern __shared__ float sharedData[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float localSum = 0.0f;
    if (idx < elementCount) {
        localSum += input[idx];
    }
    if (idx + blockDim.x < elementCount) {
        localSum += input[idx + blockDim.x];
    }
    sharedData[tid] = localSum;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

void mantener_fuentes_de_calor(float* _grid){
    if (!initialized) return;
    
    // Apply heat sources on device
    applyHeatSources<<<1, 1>>>(deviceGridCurrent, grid_size, pitchElements);
    cudaDeviceSynchronize();
    
    // Sync to host if needed
    if (_grid == grid) {
        copyPitchedToFlat(deviceGridCurrent, grid, grid_size, pitchElements);
    }
}

void initialize_grid(int N) {
    if (initialized) {
        destroy__grid();
    }
    
    grid_size = N;
    pitchElements = grid_size + PADDING_GLOBAL;
    
    // Allocate host grid
    grid = (float*)malloc(sizeof(float)*grid_size*grid_size);
    for (int i = 0; i < grid_size*grid_size; i++) {
        grid[i] = 0.0f;
    }
    
    new_grid = (float*)malloc(sizeof(float)*grid_size*grid_size);
    
    // Allocate device buffers
    size_t pitchedBytes = sizeof(float) * pitchElements * pitchElements;
    cudaMalloc(&deviceGridCurrent, pitchedBytes);
    cudaMalloc(&deviceGridNext, pitchedBytes);
    cudaMalloc(&deviceFlatBuffer, sizeof(float) * grid_size * grid_size);
    
    // Initialize device buffers to zero
    cudaMemset(deviceGridCurrent, 0, pitchedBytes);
    cudaMemset(deviceGridNext, 0, pitchedBytes);
    
    // Copy initial host grid to device
    copyFlatToPitched(grid, deviceGridCurrent, grid_size, pitchElements);
    
    // Setup reduction buffers
    int totalElements = grid_size * grid_size;
    reductionBlocks = (totalElements + THREADS_PER_BLOCK * 2 - 1) / (THREADS_PER_BLOCK * 2);
    cudaMalloc(&deviceReductionPartials, sizeof(float) * reductionBlocks);
    hostReductionPartials = (float*)malloc(sizeof(float) * reductionBlocks);
    
    // Initialize heat sources
    mantener_fuentes_de_calor(grid);
    
    previousReduction = 0.0f;
    currentReduction = 0.0f;
    initialized = true;
}

void update_simulation() {
    if (!initialized) return;
    
    // Configure kernel dimensions
    dim3 blockDim(TILE_X, (THREADS_PER_BLOCK + TILE_X - 1) / TILE_X);
    dim3 gridDim((grid_size + blockDim.x - 1) / blockDim.x,
                 (grid_size + blockDim.y - 1) / blockDim.y);
    size_t sharedBytes = (blockDim.y + 2) * (blockDim.x + 3) * sizeof(float);
    
    // 1. Diffusion step
    simulateDiffusion<<<gridDim, blockDim, sharedBytes>>>(
        deviceGridCurrent,
        deviceGridNext,
        grid_size,
        pitchElements,
        diffusion_rate);
    
    // 2. Apply heat sources
    applyHeatSources<<<1, 1>>>(deviceGridNext, grid_size, pitchElements);
    
    // 3. Copy from pitched buffer to flat buffer for reduction (handle pitch on host)
    float* tempFlatHost = (float*)malloc(sizeof(float) * grid_size * grid_size);
    copyPitchedToFlat(deviceGridNext, tempFlatHost, grid_size, pitchElements);
    cudaMemcpy(deviceFlatBuffer, tempFlatHost, sizeof(float) * grid_size * grid_size, cudaMemcpyHostToDevice);
    free(tempFlatHost);
    
    // 4. Reduction step (for convergence checking)
    cudaMemset(deviceReductionPartials, 0, sizeof(float) * reductionBlocks);
    
    int totalElements = grid_size * grid_size;
    reduceGridKernel<<<reductionBlocks, THREADS_PER_BLOCK, sizeof(float) * THREADS_PER_BLOCK>>>(
        deviceFlatBuffer,
        deviceReductionPartials,
        totalElements);
    
    cudaDeviceSynchronize();
    cudaMemcpy(hostReductionPartials, deviceReductionPartials,
               sizeof(float) * reductionBlocks, cudaMemcpyDeviceToHost);
    
    // 5. Calculate total reduction
    previousReduction = currentReduction;
    currentReduction = 0.0f;
    for (int i = 0; i < reductionBlocks; i++) {
        currentReduction += hostReductionPartials[i];
    }
    
    // 6. Swap buffers
    float* temp = deviceGridCurrent;
    deviceGridCurrent = deviceGridNext;
    deviceGridNext = temp;
    
    // 7. Sync to host grid for visualization
    copyPitchedToFlat(deviceGridCurrent, grid, grid_size, pitchElements);
}

void destroy__grid(){
    if (deviceGridCurrent) {
        cudaFree(deviceGridCurrent);
        deviceGridCurrent = NULL;
    }
    if (deviceGridNext) {
        cudaFree(deviceGridNext);
        deviceGridNext = NULL;
    }
    if (deviceFlatBuffer) {
        cudaFree(deviceFlatBuffer);
        deviceFlatBuffer = NULL;
    }
    if (deviceReductionPartials) {
        cudaFree(deviceReductionPartials);
        deviceReductionPartials = NULL;
    }
    if (hostReductionPartials) {
        free(hostReductionPartials);
        hostReductionPartials = NULL;
    }
    
    if (grid) {
        free(grid);
        grid = NULL;
    }
    if (new_grid) {
        free(new_grid);
        new_grid = NULL;
    }
    
    grid_size = 0;
    pitchElements = 0;
    reductionBlocks = 0;
    initialized = false;
}
