#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "cuda_utils.h"
#define TILE_X 32

// 2 grids para evitar conflictos de memoria en DEVICE y una en HOST para resultado final
float *d_grid, *d_grid_new, *h_grid, *tmp; 
float diffusion_rate = 0.25f;
int grid_size, pitch_elems;       


__host__ void initialize_grid(int N) {
    grid_size = N;             
    pitch_elems = N + 2;         
    h_grid = (float*)malloc(sizeof(float) * pitch_elems * pitch_elems);

    // NxN grid + padding
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
    if (x >= N || y >= N) return; 

    // Sumamos offset a punteros por padding en alocacion 
    const float* g = _grid     + pitch + 1;
    float*       o = _grid_new + pitch + 1;

 
    extern __shared__ float smem[];
    const int stride_y = blockDim.x + 3; // +2 halo + 1 padding
    const int lx = threadIdx.x + 1;
    const int ly = threadIdx.y + 1;


    size_t i = (size_t)y * pitch + x;

    // Centro del bloque (coalescente)
    smem[ly * stride_y + lx] = g[i];

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
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <grid_size> <threads_per_block> <diffusion_rate> <steps> [output_path]\n", argv[0]);
        return 1;
    }

    // Properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    PRINT_DEVICE_NAME(prop);
    PRINT_DEVICE_PROP(prop);

    // Parametrizacion de la simulacion
    int N = atoi(argv[1]);
    int threads_per_block = atoi(argv[2]);
    diffusion_rate = (float)atof(argv[3]);
    int steps = atoi(argv[4]);
    if (steps < 0) {
        fprintf(stderr, "Simulation steps must be non-negative.\n");
        return 1;
    }
    const char* output_path = (argc >= 6) ? argv[5] : "simulation_output.txt";

    // Declaración de variables de CPU y GPU
    initialize_grid(N);

    // Multiplos de 32 para aprovechar 32 bancos de memoria compartida
    dim3 block_dim(TILE_X, CEIL_DIV(threads_per_block, TILE_X));
    dim3 grid_dim(CEIL_DIV(N, block_dim.x), CEIL_DIV(N, block_dim.y));

    // Memoria compartida = Tamaño de bloque + padding (evitar conflictos de bancos) + halo (bordes de bloque)
    size_t shm_bytes = (size_t)(block_dim.y + 2) * (block_dim.x + 2 + 1) * sizeof(float);

    printf("Grid dim: %d, %d\n", grid_dim.x, grid_dim.y);
    printf("Block dim: %d, %d\n", block_dim.x, block_dim.y);
    printf("Shared memory size: %zu bytes\n", shm_bytes);
    mantener_fuentes_de_calor<<<1, 1>>>(d_grid, grid_size, pitch_elems);
    cudaDeviceSynchronize();
    double ti_total = dwalltime();
    // Iteraciones de simulación
    for (int step = 0; step < steps; ++step) {
        actualizar_simulacion<<<grid_dim, block_dim, shm_bytes>>>(d_grid, d_grid_new, N, pitch_elems, diffusion_rate);
        // Enforce sources on the freshly computed field
        mantener_fuentes_de_calor<<<1, 1>>>(d_grid_new, grid_size, pitch_elems);
        // Now make the new field the current one
        tmp = d_grid; d_grid = d_grid_new; d_grid_new = tmp;
    }
    double tf_total = dwalltime();
    printf("Time taken: %f seconds\n", tf_total - ti_total);
    cudaDeviceSynchronize();
    // Copy the entire grid (including padding) from device to host
    cudaMemcpy(h_grid, d_grid, pitch_elems * pitch_elems * sizeof(float), cudaMemcpyDeviceToHost);
    PRINT_ARRAY(h_grid, grid_size * grid_size);

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
