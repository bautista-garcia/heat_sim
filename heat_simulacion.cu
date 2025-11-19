#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "cuda_utils.h"
#define TILE_X 32
#define PADDING_SM 1
#define PADDING_GLOBAL 2


// 2 grids para evitar conflictos de memoria en DEVICE y una en HOST para resultado final
float *d_grid, *d_grid_new, *h_grid, *tmp; 
float diffusion_rate = 0.25f;
int grid_size;       


__host__ void initialize_grid(int N) {
    grid_size = N;
    // Grilla NxN + padding global          
    h_grid = (float*)malloc(sizeof(float) * N * N);

    size_t bytes = sizeof(float) * N * N;
    cudaMalloc(&d_grid, bytes);
    cudaMalloc(&d_grid_new, bytes);

    cudaMemset(d_grid, 0, bytes);
    cudaMemset(d_grid_new, 0, bytes);
}

// TODO: Kernel de unico thread para evitar transferencia entre pasos
__global__ void mantener_fuentes_de_calor(float* _grid, int grid_size){
    int cx = grid_size / 2;
    int cy = grid_size / 2;
    // Centro
    _grid[cy * grid_size + cx] = 100.0f;
    // Offsets
    int offset = 20;
    if (cy + offset < grid_size && cx + offset < grid_size) _grid[(cy + offset) * grid_size + (cx + offset)] = 100.0f;
    if (cy + offset < grid_size && cx >= offset) _grid[(cy + offset) * grid_size + (cx - offset)] = 100.0f;
    if (cy >= offset && cx + offset < grid_size) _grid[(cy - offset) * grid_size + (cx + offset)] = 100.0f;
    if (cy >= offset && cx >= offset) _grid[(cy - offset) * grid_size + (cx - offset)] = 100.0f;
}

__global__ void actualizar_simulacion(const float* __restrict__ _grid, float* __restrict__ _grid_new, int grid_size, float k) {
    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;
    // Bordes no se procesan
    if (x <= 0  ||  x >= grid_size-1  ||  y <= 0  ||  y >= grid_size-1) return; 

    // Memoria compartida
    extern __shared__ float smem[];
    const int stride = blockDim.x + PADDING_SM; // +1 padding para evitar bank conflicts
    const int lx = threadIdx.x;
    const int ly = threadIdx.y;
    // Posicion del grid a procesar (flat index)
    int i = y * grid_size + x;
    // Carga a memoria compartida
    smem[ly * stride + lx] = _grid[i]; // Centro del bloque (coalescente)
    __syncthreads();

    const int idx = ly * stride + lx;
    float c = smem[idx];
    float u = (threadIdx.y > 0 && y > 1) ? smem[(ly - 1) * stride + lx] : _grid[i - grid_size];
    float d = (threadIdx.y < blockDim.y - 1 && y < grid_size - 2) ? smem[(ly + 1) * stride + lx] : _grid[i + grid_size];
    float l = (threadIdx.x > 0 && x > 1) ? smem[ly * stride + (lx - 1)] : _grid[i - 1];
    float r = (threadIdx.x < blockDim.x - 1 && x < grid_size - 2) ? smem[ly * stride + (lx + 1)] : _grid[i + 1];
    // Escritura a memoria global
    _grid_new[i] = c + k * (u + d + l + r - 4.0f * c);
}

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

int main(int argc, char** argv){
    // Device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    PRINT_DEVICE_SUMMARY(prop);

    if (argc < 5) {
        fprintf(stderr, "Usage: %s <grid_size> <threads_per_block> <diffusion_rate> <steps> [output_path]\n", argv[0]);
        return 1;
    }

    // Parametrizacion 
    int N = atoi(argv[1]); // Tamaño de la grilla
    int threads_per_block = atoi(argv[2]);
    diffusion_rate = (float)atof(argv[3]); 
    int steps = atoi(argv[4]); // Pasos de simulacion
    if (steps < 0) {
        fprintf(stderr, "Simulation steps must be non-negative.\n");
        return 1;
    }
    const char* output_path = (argc >= 6) ? argv[5] : "simulation_output.txt";

    // Inicializacion de la grilla (con padding)
    initialize_grid(N);

    // Config de sim_step
    dim3 block_dim(TILE_X, CEIL_DIV(threads_per_block, TILE_X));
    dim3 grid_dim(CEIL_DIV(N, block_dim.x), CEIL_DIV(N, block_dim.y));
    // Memoria compartida = Tamaño de bloque + padding (evitar conflictos de bancos) + halo (bordes de bloque)
    size_t shm_bytes = (size_t)(block_dim.y) * (block_dim.x + PADDING_SM) * sizeof(float);

    // Config de reduccion
    int num_blocks_reduce = CEIL_DIV(N * N, threads_per_block * 2); // 2 because of the first reduction step
    int shm_bytes_reduce = threads_per_block * sizeof(float);
    float *d_result_reduce;
    cudaMalloc(&d_result_reduce, sizeof(float) * num_blocks_reduce);
    cudaMemset(d_result_reduce, 0, sizeof(float) * num_blocks_reduce);
    float *h_result_reduce = (float*)malloc(sizeof(float) * num_blocks_reduce);

    printf("\nReduction configuration:\n");
    printf("Total elements: %d\n", N * N);
    printf("Threads per block: %d\n", threads_per_block);
    printf("Number of blocks: %d\n", num_blocks_reduce);
    printf("Shared memory per block: %d bytes\n", shm_bytes_reduce);


    // Dimensiones con parametros ingresados
    printf("\nSimulation configuration:\n");
    printf("Grid dim: %d, %d\n", grid_dim.x, grid_dim.y);
    printf("Block dim: %d, %d\n", block_dim.x, block_dim.y);
    printf("Shared memory size: %zu bytes\n", shm_bytes);

    // // Occupancy calculation
    // int threads_per_block_total = block_dim.x * block_dim.y;
    // int numBlocks;
    // cudaError_t occupancyErr = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocks,
    //     actualizar_simulacion,
    //     threads_per_block_total,
    //     shm_bytes
    // );
    
    // if (occupancyErr != cudaSuccess) {
    //     fprintf(stderr, "Error calculating occupancy: %s\n", cudaGetErrorString(occupancyErr));
    // } else {
    //     int warp_size = prop.warpSize;
    //     // Calculate max threads per multiprocessor based on compute capability
    //     int max_threads_per_sm;
    //     if (prop.major == 1) {
    //         max_threads_per_sm = 768;
    //     } else if (prop.major == 2) {
    //         max_threads_per_sm = 1536;
    //     } else {
    //         // Compute capability 3.0 and higher
    //         max_threads_per_sm = 2048;
    //     }
        
    //     int max_warps_per_sm = max_threads_per_sm / warp_size;
    //     int warps_per_block = (threads_per_block_total + warp_size - 1) / warp_size;
    //     int active_warps_per_sm = numBlocks * warps_per_block;
    //     double occupancy_percent = (double)active_warps_per_sm / max_warps_per_sm * 100.0;
        
    //     printf("\n======================= OCCUPANCY ANALYSIS =======================\n");
    //     printf("Threads per block: %d\n", threads_per_block_total);
    //     printf("Warps per block: %d\n", warps_per_block);
    //     printf("Max active blocks per SM: %d\n", numBlocks);
    //     printf("Active warps per SM: %d / %d\n", active_warps_per_sm, max_warps_per_sm);
    //     printf("Occupancy: %.2f%%\n", occupancy_percent);
    //     printf("Shared memory per block: %zu bytes (limit: %zu bytes)\n", 
    //            shm_bytes, (size_t)prop.sharedMemPerBlock);
    //     printf("Shared memory per SM: %zu bytes (limit: %zu bytes)\n",
    //            (size_t)(numBlocks * shm_bytes), (size_t)prop.sharedMemPerMultiprocessor);
    //     printf("==============================================================\n\n");
    // }

    // Simulacion
    float result_reduce_prev = 0.0f, result_reduce_curr = 0.0f;
    mantener_fuentes_de_calor<<<1, 1>>>(d_grid, grid_size);
    cudaDeviceSynchronize();
    double ti_total = dwalltime();
    for (int step = 0; step < steps; ++step) {
        // PASO DE SIMULACION
        actualizar_simulacion<<<grid_dim, block_dim, shm_bytes>>>(d_grid, d_grid_new, grid_size, diffusion_rate);
        mantener_fuentes_de_calor<<<1, 1>>>(d_grid_new, grid_size);
        result_reduce_prev = result_reduce_curr;
        // PASO DE REDUCCION
        cudaMemset(d_result_reduce, 0, sizeof(float) * num_blocks_reduce);
        reduction_kernel<<<num_blocks_reduce, threads_per_block, shm_bytes_reduce>>>(d_grid_new, d_result_reduce, grid_size * grid_size);
        cudaDeviceSynchronize();
        cudaMemcpy(h_result_reduce, d_result_reduce, sizeof(float) * num_blocks_reduce, cudaMemcpyDeviceToHost);
        float result_reduce = 0.0f;
        for (int i = 0; i < num_blocks_reduce; i++) {
            result_reduce += h_result_reduce[i];
        }
        result_reduce_curr = result_reduce;
        if (result_reduce_curr - result_reduce_prev < 0.0001f) {
            printf("Result reduce diff is less than 0.0001f, stopping simulation at step %d\n", step);
            break;
        }
        // Swap de punteros
        tmp = d_grid; d_grid = d_grid_new; d_grid_new = tmp;
    }
    double tf_total = dwalltime();
    printf("Time taken: %f seconds\n", tf_total - ti_total);

    cudaDeviceSynchronize();
    cudaMemcpy(h_grid, d_grid, grid_size * grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    // --------------------------------------------------------------
    // VISUALIZACION CON PYTHON (EXTRA)
    FILE* fp = fopen(output_path, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open output file '%s' for writing.\n", output_path);
        free(h_grid);
        cudaFree(d_grid);
        cudaFree(d_grid_new);
        return 1;
    }

    fprintf(fp, "%d\n", grid_size);    // Plain grid, no pitch/padding
    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            fprintf(fp, "%.6f", h_grid[y * grid_size + x]);
            if (x < grid_size - 1) {
                fputc(' ', fp);
            }
        }
        fputc('\n', fp);
    }
    fclose(fp);
    printf("Saved simulation snapshot to %s\n", output_path);
    // --------------------------------------------------------------

    free(h_grid);
    free(h_result_reduce);
    cudaFree(d_grid);
    cudaFree(d_grid_new);
    return 0;
}
