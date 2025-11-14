#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <math.h>
#include <utility>
#include "cuda_utils.h"
#include "heat_simulation.h"

#define TILE_X 32
#define PADDING_SM 1
#define PADDING_GLOBAL 2
#define HALO_SM 2
#define REDUCTION_STOP_THRESHOLD 0.0001f

// Default parameters
#define DEFAULT_THREADS_PER_BLOCK 256
#define DEFAULT_DIFFUSION_RATE 0.25f

// ============================================================================
// GLOBAL VARIABLES (matching heat_simulation.h interface)
// ============================================================================

float* grid = NULL;
int grid_size = 0;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

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

// Internal state
static SimulationBuffers g_simBuffers{};
static ReductionBuffers g_reductionBuffers{};
static KernelConfig g_kernelConfig{};
static float g_diffusionRate = DEFAULT_DIFFUSION_RATE;
static int g_threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
static float g_previousReduction = 0.0f;
static float g_currentReduction = 0.0f;
static bool g_initialized = false;

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

__global__ void reduceGridKernelPitched(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int gridSize,
                                        int pitchElements,
                                        int elementCount) {
    extern __shared__ float sharedData[];

    unsigned int tid = threadIdx.x;
    unsigned int linearIdx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float localSum = 0.0f;
    int baseOffset = pitchElements + 1;
    
    if (linearIdx < elementCount) {
        int y = linearIdx / gridSize;
        int x = linearIdx % gridSize;
        int idx = baseOffset + y * pitchElements + x;
        localSum += input[idx];
    }
    if (linearIdx + blockDim.x < elementCount) {
        int y = (linearIdx + blockDim.x) / gridSize;
        int x = (linearIdx + blockDim.x) % gridSize;
        int idx = baseOffset + y * pitchElements + x;
        localSum += input[idx];
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

// ============================================================================
// HOST GRID SYNC FUNCTION
// ============================================================================

static void syncHostGrid() {
    if (!g_initialized || !grid) return;
    
    cudaDeviceSynchronize();
    int baseOffset = g_simBuffers.pitchElements + 1;
    
    // Copy from device to host (full padded buffer)
    size_t bytesToCopy = sizeof(float) * g_simBuffers.pitchElements * g_simBuffers.pitchElements;
    cudaMemcpy(g_simBuffers.hostGrid,
               g_simBuffers.deviceGridCurrent,
               bytesToCopy,
               cudaMemcpyDeviceToHost);
    
    // Copy from padded host buffer to flat grid array
    for (int y = 0; y < grid_size; ++y) {
        for (int x = 0; x < grid_size; ++x) {
            grid[y * grid_size + x] = g_simBuffers.hostGrid[baseOffset + y * g_simBuffers.pitchElements + x];
        }
    }
}

// ============================================================================
// INTERFACE FUNCTIONS (matching heat_simulation.h)
// ============================================================================

void mantener_fuentes_de_calor(float* _grid) {
    // This function is mainly for compatibility with the interface
    // The actual heat sources are applied on the GPU
    // If _grid is the host grid, we need to sync it to device first
    if (!g_initialized) return;
    
    // Apply heat sources on device
    applyHeatSources<<<1, 1>>>(
        g_simBuffers.deviceGridCurrent,
        grid_size,
        g_simBuffers.pitchElements);
    cudaDeviceSynchronize();
    
    // If _grid is provided and different from our grid, update it
    if (_grid && _grid != grid) {
        syncHostGrid();
        for (int i = 0; i < grid_size * grid_size; ++i) {
            _grid[i] = grid[i];
        }
    }
}

void initialize_grid(int N) {
    if (g_initialized) {
        destroy__grid();
    }
    
    grid_size = N;
    
    // Allocate host grid (flat array for visualization)
    grid = (float*)malloc(sizeof(float) * grid_size * grid_size);
    for (int i = 0; i < grid_size * grid_size; i++) {
        grid[i] = 0.0f;
    }
    
    // Allocate GPU buffers
    g_simBuffers = allocateSimulationBuffers(grid_size);
    
    // Configure kernel dimensions
    int totalElements = grid_size * grid_size;
    g_kernelConfig.simulationBlockDim = dim3(TILE_X, CEIL_DIV(g_threadsPerBlock, TILE_X));
    g_kernelConfig.simulationGridDim = dim3(CEIL_DIV(grid_size, g_kernelConfig.simulationBlockDim.x),
                                            CEIL_DIV(grid_size, g_kernelConfig.simulationBlockDim.y));
    g_kernelConfig.simulationSharedBytes = static_cast<size_t>(g_kernelConfig.simulationBlockDim.y + HALO_SM)
                                         * (g_kernelConfig.simulationBlockDim.x + HALO_SM + PADDING_SM) * sizeof(float);
    g_kernelConfig.reductionBlocks = CEIL_DIV(totalElements, g_threadsPerBlock * 2);
    g_kernelConfig.reductionSharedBytes = static_cast<size_t>(g_threadsPerBlock) * sizeof(float);
    
    // Allocate reduction buffers
    g_reductionBuffers = allocateReductionBuffers(g_kernelConfig.reductionBlocks);
    
    // Initialize heat sources
    mantener_fuentes_de_calor(NULL);
    
    // Sync initial state to host
    syncHostGrid();
    
    g_previousReduction = 0.0f;
    g_currentReduction = 0.0f;
    g_initialized = true;
}

void update_simulation() {
    if (!g_initialized) return;
    
    // Unified simulation step: diffusion + heat sources + reduction check
    
    // 1. Diffusion step
    simulateDiffusion<<<g_kernelConfig.simulationGridDim,
                        g_kernelConfig.simulationBlockDim,
                        g_kernelConfig.simulationSharedBytes>>>(
        g_simBuffers.deviceGridCurrent,
        g_simBuffers.deviceGridNext,
        grid_size,
        g_simBuffers.pitchElements,
        g_diffusionRate);
    
    // 2. Apply heat sources
    applyHeatSources<<<1, 1>>>(
        g_simBuffers.deviceGridNext,
        grid_size,
        g_simBuffers.pitchElements);
    
    // 3. Reduction step (for convergence checking)
    size_t reductionBytes = sizeof(float) * g_kernelConfig.reductionBlocks;
    cudaMemset(g_reductionBuffers.devicePartials, 0, reductionBytes);
    
    int totalElements = grid_size * grid_size;
    reduceGridKernelPitched<<<g_kernelConfig.reductionBlocks,
                              g_threadsPerBlock,
                              g_kernelConfig.reductionSharedBytes>>>(
        g_simBuffers.deviceGridNext,
        g_reductionBuffers.devicePartials,
        grid_size,
        g_simBuffers.pitchElements,
        totalElements);
    
    cudaDeviceSynchronize();
    cudaMemcpy(g_reductionBuffers.hostPartials,
               g_reductionBuffers.devicePartials,
               reductionBytes,
               cudaMemcpyDeviceToHost);
    
    // 4. Calculate total reduction
    g_previousReduction = g_currentReduction;
    g_currentReduction = 0.0f;
    for (int i = 0; i < g_kernelConfig.reductionBlocks; ++i) {
        g_currentReduction += g_reductionBuffers.hostPartials[i];
    }
    
    // 5. Swap buffers for next iteration
    std::swap(g_simBuffers.deviceGridCurrent, g_simBuffers.deviceGridNext);
    
    // 6. Sync host grid for visualization
    syncHostGrid();
}

void destroy__grid() {
    if (!g_initialized) return;
    
    releaseSimulationBuffers(g_simBuffers);
    releaseReductionBuffers(g_reductionBuffers);
    
    if (grid) {
        free(grid);
        grid = NULL;
    }
    
    grid_size = 0;
    g_initialized = false;
}

// ============================================================================
// MAIN FUNCTION (for standalone execution)
// ============================================================================

#ifdef STANDALONE_MAIN
int main(int argc, char** argv) {
    // Device properties
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);
    PRINT_DEVICE_SUMMARY(deviceProperties);

    if (argc < 5) {
        fprintf(stderr, "Usage: %s <grid_size> <threads_per_block> <diffusion_rate> <steps> [output_path]\n", argv[0]);
        return 1;
    }

    int gridSize = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    float diffusionRate = static_cast<float>(atof(argv[3]));
    int maxSteps = atoi(argv[4]);
    const char* outputPath = (argc >= 6) ? argv[5] : "simulation_output.txt";

    // Parameter validation
    if (gridSize <= 0) {
        fprintf(stderr, "Grid size must be positive.\n");
        return 1;
    }
    if (threadsPerBlock <= 0) {
        fprintf(stderr, "Threads per block must be positive.\n");
        return 1;
    }
    if (maxSteps < 0) {
        fprintf(stderr, "Simulation steps must be non-negative.\n");
        return 1;
    }

    // Set internal parameters
    g_threadsPerBlock = threadsPerBlock;
    g_diffusionRate = diffusionRate;

    printf("\n======================= PARAMETERS =======================\n");
    printf("Grid size: %d\n", gridSize);
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Diffusion rate: %.4f\n", diffusionRate);
    printf("Max steps: %d\n", maxSteps);
    printf("Output path: %s\n", outputPath);
    printf("==========================================================\n");

    // Initialize using interface function
    initialize_grid(gridSize);

    // Run simulation
    double simulationStart = dwalltime();
    for (int step = 0; step < maxSteps; ++step) {
        update_simulation();
        
        // Check convergence (optional - can be removed for continuous simulation)
        if (step > 0 && fabsf(g_currentReduction - g_previousReduction) < REDUCTION_STOP_THRESHOLD) {
            printf("Result reduction diff below %.5f, stopping at step %d\n", REDUCTION_STOP_THRESHOLD, step);
            break;
        }
    }
    double simulationEnd = dwalltime();

    // Write output
    FILE* fp = fopen(outputPath, "w");
    if (fp) {
        fprintf(fp, "%d\n", gridSize);
        for (int y = 0; y < gridSize; ++y) {
            for (int x = 0; x < gridSize; ++x) {
                fprintf(fp, "%.6f", grid[y * gridSize + x]);
                if (x < gridSize - 1) {
                    fputc(' ', fp);
                }
            }
            fputc('\n', fp);
        }
        fclose(fp);
        printf("Saved simulation snapshot to %s\n", outputPath);
    }

    printf("\n================== RESULTS AND TIMINGS ===================\n");
    printf("Simulation time: %.6f seconds\n", simulationEnd - simulationStart);
    printf("Final reduction value: %.6f\n", g_currentReduction);
    printf("==========================================================\n");

    // Cleanup
    destroy__grid();

    return 0;
}
#endif
