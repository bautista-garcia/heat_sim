#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "cuda_utils.h"
#define TILE_X 32
#define PADDING_SM 1
#define PADDING_GLOBAL 2
#define HALO_SM 2


// 2 grids para evitar conflictos de memoria en DEVICE y una en HOST para resultado final
float *d_grid, *d_grid_new, *h_grid, *tmp; 
float diffusion_rate = 0.25f;
int grid_size, pitch_elems;       


__host__ void initialize_grid(int N) {
    // Grilla NxN + padding global
    grid_size = N;             
    pitch_elems = N + PADDING_GLOBAL;         
    h_grid = (float*)malloc(sizeof(float) * pitch_elems * pitch_elems);

    size_t bytes = sizeof(float) * pitch_elems * pitch_elems;
    cudaMalloc(&d_grid, bytes);
    cudaMalloc(&d_grid_new, bytes);

    cudaMemset(d_grid, 0, bytes);
    cudaMemset(d_grid_new, 0, bytes);
}

// TODO: Kernel de unico thread para evitar transferencia entre pasos
__global__ void mantener_fuentes_de_calor(float* _grid, int grid_size, int pitch){
    int cx = grid_size / 2;
    int cy = grid_size / 2;
    
    // Account for padding: data starts at (pitch + 1)
    int base_offset = pitch + 1;

    _grid[base_offset + cy * pitch + cx] = 100.0f;

    int offset = 8;

    if (cy + offset < grid_size && cx + offset < grid_size) {
        _grid[base_offset + (cy + offset) * pitch + (cx + offset)] = 100.0f;
    }
    if (cy + offset < grid_size && cx >= offset) {
        _grid[base_offset + (cy + offset) * pitch + (cx - offset)] = 100.0f;
    }
    if (cy >= offset && cx + offset < grid_size) {
        _grid[base_offset + (cy - offset) * pitch + (cx + offset)] = 100.0f;
    }
    if (cy >= offset && cx >= offset) {
        _grid[base_offset + (cy - offset) * pitch + (cx - offset)] = 100.0f;
    }
}

__global__ void actualizar_simulacion(const float* __restrict__ _grid, float* __restrict__ _grid_new, int N, int pitch, float k) {
    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;
    // Bordes no se procesan
    if (x >= N ||  y >= N) return; 
    // Sumamos offset a punteros por padding en alocacion 
    const float* g = _grid     + pitch + 1;
    float*       o = _grid_new + pitch + 1;
    // Memoria compartida
    extern __shared__ float smem[];
    const int stride_y = blockDim.x + 3; // +2 halo + 1 padding
    const int lx = threadIdx.x + 1;
    const int ly = threadIdx.y + 1;
    // Posicion del grid a procesar (flat index)
    int i = y * pitch + x;

    // Carga a memoria compartida
    smem[ly * stride_y + lx] = g[i]; // Centro del bloque (coalescente)
    if (threadIdx.y == 0) smem[lx] = g[i - pitch];                 // Halo superior (coalescente)
    if (threadIdx.y == blockDim.y - 1) smem[(blockDim.y + 1) * stride_y + lx] = g[i + pitch]; // Halo inferior (coalescente)
    if (threadIdx.x == 0) smem[ly * stride_y] = g[i - 1];         // Halo izquierdo (no coalescente)
    if (threadIdx.x == blockDim.x - 1) smem[ly * stride_y + (blockDim.x + 1)] = g[i + 1];     // Halo derecho (no coalescente)

    __syncthreads();

    float c = smem[ly * stride_y + lx];
    float u = smem[(ly-1) * stride_y + lx];
    float d = smem[(ly+1) * stride_y + lx];
    float l = smem[ly * stride_y + (lx-1)];
    float r = smem[ly * stride_y + (lx+1)];

    o[i] = c + k * (u + d + l + r - 4.0f * c);
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

    // Multiplos de 32 para aprovechar 32 bancos de memoria compartida
    dim3 block_dim(TILE_X, CEIL_DIV(threads_per_block, TILE_X));
    dim3 grid_dim(CEIL_DIV(N, block_dim.x), CEIL_DIV(N, block_dim.y));
    // Memoria compartida = Tamaño de bloque + padding (evitar conflictos de bancos) + halo (bordes de bloque)
    size_t shm_bytes = (size_t)(block_dim.y + HALO_SM) * (block_dim.x + HALO_SM + PADDING_SM) * sizeof(float);

    // Dimensiones con parametros ingresados
    printf("Grid dim: %d, %d\n", grid_dim.x, grid_dim.y);
    printf("Block dim: %d, %d\n", block_dim.x, block_dim.y);
    printf("Shared memory size: %zu bytes\n", shm_bytes);

    // Occupancy calculation
    int threads_per_block_total = block_dim.x * block_dim.y;
    int numBlocks;
    cudaError_t occupancyErr = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        actualizar_simulacion,
        threads_per_block_total,
        shm_bytes
    );
    
    if (occupancyErr != cudaSuccess) {
        fprintf(stderr, "Error calculating occupancy: %s\n", cudaGetErrorString(occupancyErr));
    } else {
        int warp_size = prop.warpSize;
        // Calculate max threads per multiprocessor based on compute capability
        int max_threads_per_sm;
        if (prop.major == 1) {
            max_threads_per_sm = 768;
        } else if (prop.major == 2) {
            max_threads_per_sm = 1536;
        } else {
            // Compute capability 3.0 and higher
            max_threads_per_sm = 2048;
        }
        
        int max_warps_per_sm = max_threads_per_sm / warp_size;
        int warps_per_block = (threads_per_block_total + warp_size - 1) / warp_size;
        int active_warps_per_sm = numBlocks * warps_per_block;
        double occupancy_percent = (double)active_warps_per_sm / max_warps_per_sm * 100.0;
        
        printf("\n======================= OCCUPANCY ANALYSIS =======================\n");
        printf("Threads per block: %d\n", threads_per_block_total);
        printf("Warps per block: %d\n", warps_per_block);
        printf("Max active blocks per SM: %d\n", numBlocks);
        printf("Active warps per SM: %d / %d\n", active_warps_per_sm, max_warps_per_sm);
        printf("Occupancy: %.2f%%\n", occupancy_percent);
        printf("Shared memory per block: %zu bytes (limit: %zu bytes)\n", 
               shm_bytes, (size_t)prop.sharedMemPerBlock);
        printf("Shared memory per SM: %zu bytes (limit: %zu bytes)\n",
               (size_t)(numBlocks * shm_bytes), (size_t)prop.sharedMemPerMultiprocessor);
        printf("==============================================================\n\n");
    }

    // Simulacion
    mantener_fuentes_de_calor<<<1, 1>>>(d_grid, grid_size, pitch_elems);
    cudaDeviceSynchronize();
    double ti_total = dwalltime();
    for (int step = 0; step < steps; ++step) {
        actualizar_simulacion<<<grid_dim, block_dim, shm_bytes>>>(d_grid, d_grid_new, N, pitch_elems, diffusion_rate);
        mantener_fuentes_de_calor<<<1, 1>>>(d_grid_new, grid_size, pitch_elems);
        // Swap de punteros
        tmp = d_grid; d_grid = d_grid_new; d_grid_new = tmp;
    }
    double tf_total = dwalltime();
    printf("Time taken: %f seconds\n", tf_total - ti_total);

    cudaDeviceSynchronize();
    cudaMemcpy(h_grid, d_grid, pitch_elems * pitch_elems * sizeof(float), cudaMemcpyDeviceToHost);

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

    fprintf(fp, "%d\n", grid_size);
    // Extract data region from padded grid: data starts at (pitch_elems + 1)
    int base_offset = pitch_elems + 1;
    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            fprintf(fp, "%.6f", h_grid[base_offset + y * pitch_elems + x]);
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
    cudaFree(d_grid);
    cudaFree(d_grid_new);
    return 0;
}
