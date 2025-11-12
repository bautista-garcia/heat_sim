#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <math.h>
#include <utility>
#include "cuda_utils.h"

#define TILE_X 32
#define PADDING_SM 1
#define PADDING_GLOBAL 2
#define HALO_SM 2
#define REDUCTION_STOP_THRESHOLD 0.0001f

// ============================================================================
// DATA STRUCTURES
// ============================================================================

struct SimulationParameters {
    int gridSize{};
    int threadsPerBlock{};
    float diffusionRate{};
    int maxSteps{};
    int totalElements{};
    const char* outputPath{};
};

struct KernelConfig {
    dim3 simulationBlockDim{};
    dim3 simulationGridDim{};
    size_t simulationSharedBytes{};
    int reductionBlocks{};
    size_t reductionSharedBytes{};
};

struct SimulationBuffers {
    float* deviceGridCurrent{nullptr};
    float* deviceGridNext{nullptr};
    float* hostGrid{nullptr};
    int pitchElements{};
};

struct ReductionBuffers {
    float* devicePartials{nullptr};
    float* hostPartials{nullptr};
};

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

__host__ SimulationBuffers allocateSimulationBuffers(int gridSize) {
    SimulationBuffers buffers{};
    buffers.pitchElements = gridSize + PADDING_GLOBAL;
    size_t totalBytes = sizeof(float) * buffers.pitchElements * buffers.pitchElements;

    buffers.hostGrid = static_cast<float*>(malloc(totalBytes));
    cudaMalloc(&buffers.deviceGridCurrent, totalBytes);
    cudaMalloc(&buffers.deviceGridNext, totalBytes);

    cudaMemset(buffers.deviceGridCurrent, 0, totalBytes);
    cudaMemset(buffers.deviceGridNext, 0, totalBytes);

    return buffers;
}

__host__ void releaseSimulationBuffers(SimulationBuffers& buffers) {
    if (buffers.hostGrid) {
        free(buffers.hostGrid);
        buffers.hostGrid = nullptr;
    }
    if (buffers.deviceGridCurrent) {
        cudaFree(buffers.deviceGridCurrent);
        buffers.deviceGridCurrent = nullptr;
    }
    if (buffers.deviceGridNext) {
        cudaFree(buffers.deviceGridNext);
        buffers.deviceGridNext = nullptr;
    }
    buffers.pitchElements = 0;
}

__host__ ReductionBuffers allocateReductionBuffers(int reductionBlocks) {
    ReductionBuffers buffers{};
    size_t bytes = sizeof(float) * reductionBlocks;

    cudaMalloc(&buffers.devicePartials, bytes);
    cudaMemset(buffers.devicePartials, 0, bytes);

    buffers.hostPartials = static_cast<float*>(malloc(bytes));
    return buffers;
}

__host__ void releaseReductionBuffers(ReductionBuffers& buffers) {
    if (buffers.devicePartials) {
        cudaFree(buffers.devicePartials);
        buffers.devicePartials = nullptr;
    }
    if (buffers.hostPartials) {
        free(buffers.hostPartials);
        buffers.hostPartials = nullptr;
    }
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

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

double dwalltime() {
    double seconds;
    struct timeval tv;

    gettimeofday(&tv, NULL);
    seconds = tv.tv_sec + tv.tv_usec / 1000000.0;
    return seconds;
}

bool writeSimulationSnapshot(const SimulationParameters& params,
                             const SimulationBuffers& buffers) {
    FILE* fp = fopen(params.outputPath, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open output file '%s' for writing.\n", params.outputPath);
        return false;
    }

    fprintf(fp, "%d\n", params.gridSize);
    int baseOffset = buffers.pitchElements + 1;
    for (int y = 0; y < params.gridSize; ++y) {
        for (int x = 0; x < params.gridSize; ++x) {
            fprintf(fp, "%.6f", buffers.hostGrid[baseOffset + y * buffers.pitchElements + x]);
            if (x < params.gridSize - 1) {
                fputc(' ', fp);
            }
        }
        fputc('\n', fp);
    }

    fclose(fp);
    printf("Saved simulation snapshot to %s\n", params.outputPath);
    return true;
}

// ============================================================================
// SIMULATION STEP FUNCTION
// ============================================================================

bool executeSimulationStep(int step,
                           const SimulationParameters& params,
                           const KernelConfig& kernelConfig,
                           SimulationBuffers& simBuffers,
                           ReductionBuffers& reductionBuffers,
                           float& previousReduction,
                           float& currentReduction) {
    // Diffusion step
    simulateDiffusion<<<kernelConfig.simulationGridDim,
                        kernelConfig.simulationBlockDim,
                        kernelConfig.simulationSharedBytes>>>(
        simBuffers.deviceGridCurrent,
        simBuffers.deviceGridNext,
        params.gridSize,
        simBuffers.pitchElements,
        params.diffusionRate);

    // Apply heat sources
    applyHeatSources<<<1, 1>>>(
        simBuffers.deviceGridNext,
        params.gridSize,
        simBuffers.pitchElements);

    // Reduction step
    size_t reductionBytes = sizeof(float) * kernelConfig.reductionBlocks;
    cudaMemset(reductionBuffers.devicePartials, 0, reductionBytes);

    reduceGridKernel<<<kernelConfig.reductionBlocks,
                       params.threadsPerBlock,
                       kernelConfig.reductionSharedBytes>>>(
        simBuffers.deviceGridNext,
        reductionBuffers.devicePartials,
        params.totalElements);

    cudaDeviceSynchronize();
    cudaMemcpy(reductionBuffers.hostPartials,
               reductionBuffers.devicePartials,
               reductionBytes,
               cudaMemcpyDeviceToHost);

    // Calculate total reduction
    previousReduction = currentReduction;
    currentReduction = 0.0f;
    for (int i = 0; i < kernelConfig.reductionBlocks; ++i) {
        currentReduction += reductionBuffers.hostPartials[i];
    }

    // Check convergence
    if (fabsf(currentReduction - previousReduction) < REDUCTION_STOP_THRESHOLD) {
        printf("Result reduction diff below %.5f, stopping at step %d\n", REDUCTION_STOP_THRESHOLD, step);
        return false;
    }

    // Swap buffers for next iteration
    std::swap(simBuffers.deviceGridCurrent, simBuffers.deviceGridNext);
    return true;
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char** argv) {
    // Device properties
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);
    PRINT_DEVICE_SUMMARY(deviceProperties);

    if (argc < 5) {
        fprintf(stderr, "Usage: %s <grid_size> <threads_per_block> <diffusion_rate> <steps> [output_path]\n", argv[0]);
        return 1;
    }

    // ========================================================================
    // PARAMETERS
    // ========================================================================
    SimulationParameters params{};
    params.gridSize = atoi(argv[1]);
    params.threadsPerBlock = atoi(argv[2]);
    params.diffusionRate = static_cast<float>(atof(argv[3]));
    params.maxSteps = atoi(argv[4]);
    params.outputPath = (argc >= 6) ? argv[5] : "simulation_output.txt";

    // Parameter validation
    if (params.gridSize <= 0) {
        fprintf(stderr, "Grid size must be positive.\n");
        return 1;
    }
    if (params.threadsPerBlock <= 0) {
        fprintf(stderr, "Threads per block must be positive.\n");
        return 1;
    }
    if (params.maxSteps < 0) {
        fprintf(stderr, "Simulation steps must be non-negative.\n");
        return 1;
    }

    params.totalElements = params.gridSize * params.gridSize;

    printf("\n======================= PARAMETERS =======================\n");
    printf("Grid size: %d\n", params.gridSize);
    printf("Threads per block: %d\n", params.threadsPerBlock);
    printf("Diffusion rate: %.4f\n", params.diffusionRate);
    printf("Max steps: %d\n", params.maxSteps);
    printf("Output path: %s\n", params.outputPath);
    printf("Total elements: %d\n", params.totalElements);
    printf("==========================================================\n");

    // Allocate buffers
    SimulationBuffers simBuffers = allocateSimulationBuffers(params.gridSize);

    // ========================================================================
    // KERNEL DIMENSION CONFIGURATION
    // ========================================================================
    KernelConfig kernelConfig{};
    kernelConfig.simulationBlockDim = dim3(TILE_X, CEIL_DIV(params.threadsPerBlock, TILE_X));
    kernelConfig.simulationGridDim = dim3(CEIL_DIV(params.gridSize, kernelConfig.simulationBlockDim.x),
                                          CEIL_DIV(params.gridSize, kernelConfig.simulationBlockDim.y));
    kernelConfig.simulationSharedBytes = static_cast<size_t>(kernelConfig.simulationBlockDim.y + HALO_SM)
                                       * (kernelConfig.simulationBlockDim.x + HALO_SM + PADDING_SM) * sizeof(float);
    kernelConfig.reductionBlocks = CEIL_DIV(params.totalElements, params.threadsPerBlock * 2);
    kernelConfig.reductionSharedBytes = static_cast<size_t>(params.threadsPerBlock) * sizeof(float);

    ReductionBuffers reductionBuffers = allocateReductionBuffers(kernelConfig.reductionBlocks);

    printf("\n============ KERNEL DIMENSION CONFIGURATION =============\n");
    printf("Simulation grid dim: %d x %d\n", kernelConfig.simulationGridDim.x, kernelConfig.simulationGridDim.y);
    printf("Simulation block dim: %d x %d\n", kernelConfig.simulationBlockDim.x, kernelConfig.simulationBlockDim.y);
    printf("Simulation shared memory: %zu bytes\n", kernelConfig.simulationSharedBytes);
    printf("Reduction blocks: %d\n", kernelConfig.reductionBlocks);
    printf("Reduction shared memory: %zu bytes\n", kernelConfig.reductionSharedBytes);
    printf("==========================================================\n");

    // Initialize heat sources
    applyHeatSources<<<1, 1>>>(
        simBuffers.deviceGridCurrent,
        params.gridSize,
        simBuffers.pitchElements);
    cudaDeviceSynchronize();

    // ========================================================================
    // SIMULATION
    // ========================================================================
    float previousReduction = 0.0f;
    float currentReduction = 0.0f;

    double simulationStart = dwalltime();
    for (int step = 0; step < params.maxSteps; ++step) {
        bool shouldContinue = executeSimulationStep(step,
                                                    params,
                                                    kernelConfig,
                                                    simBuffers,
                                                    reductionBuffers,
                                                    previousReduction,
                                                    currentReduction);
        if (!shouldContinue) {
            break;
        }
    }
    double simulationEnd = dwalltime();

    // ========================================================================
    // RESULTS AND TIMINGS
    // ========================================================================
    cudaDeviceSynchronize();
    size_t bytesToCopy = sizeof(float) * simBuffers.pitchElements * simBuffers.pitchElements;
    cudaMemcpy(simBuffers.hostGrid,
               simBuffers.deviceGridCurrent,
               bytesToCopy,
               cudaMemcpyDeviceToHost);

    printf("\n================== RESULTS AND TIMINGS ===================\n");
    printf("Simulation time: %.6f seconds\n", simulationEnd - simulationStart);
    printf("Final reduction value: %.6f\n", currentReduction);
    printf("==========================================================\n");

    if (!writeSimulationSnapshot(params, simBuffers)) {
        releaseSimulationBuffers(simBuffers);
        releaseReductionBuffers(reductionBuffers);
        return 1;
    }

    // ========================================================================
    // DEALLOCATION
    // ========================================================================
    releaseSimulationBuffers(simBuffers);
    releaseReductionBuffers(reductionBuffers);

    return 0;
}
