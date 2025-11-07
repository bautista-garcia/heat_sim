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
int grid_size = 0;


__host__ void initialize_grid(int N) {
    grid_size = N;
    // Host grid
    h_grid = (float*)malloc(sizeof(float)*grid_size*grid_size);

    // Device grids
    cudaMalloc(&d_grid, grid_size * grid_size * sizeof(float));
    cudaMalloc(&d_grid_new, grid_size * grid_size * sizeof(float));
    cudaMemset(d_grid, 0, grid_size * grid_size * sizeof(float));  
    cudaMemset(d_grid_new, 0, grid_size * grid_size * sizeof(float));  
}

// TODO: Kernel de unico thread para evitar transferencia entre pasos
__global__ void mantener_fuentes_de_calor(float* _grid, int grid_size){
    int cx = grid_size / 2;
    int cy = grid_size / 2;

    _grid[cy * grid_size + cx] = 100.0f;

    int offset = 8;

    if (cy + offset < grid_size && cx + offset < grid_size) {
        _grid[(cy + offset) * grid_size + (cx + offset)] = 100.0f;
    }
    if (cy + offset < grid_size && cx >= offset) {
        _grid[(cy + offset) * grid_size + (cx - offset)] = 100.0f;
    }
    if (cy >= offset && cx + offset < grid_size) {
        _grid[(cy - offset) * grid_size + (cx + offset)] = 100.0f;
    }
    if (cy >= offset && cx >= offset) {
        _grid[(cy - offset) * grid_size + (cx - offset)] = 100.0f;
    }
}

__global__ void actualizar_simulacion(float* _grid, float* _grid_new, int N, float k){
    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // ---- Dynamic shared memory with +1 padding in X ----
    extern __shared__ float smem[];
    const int pitch = blockDim.x + 1;                    // +1 padding in X
    float* tile = smem;                                  // [blockDim.y][blockDim.x+1] linearized

    if (x >= N || y >= N) return;

    // Etapa 1: Acceso coalescente + memoria compartida
    tile[ty * pitch + tx] = _grid[y * N + x];
    __syncthreads();

    // Etapa 2: Calculo de la temperatura (interior del bloque y del dominio)
    if (x > 0 && x < N-1 && y > 0 && y < N-1 &&
        tx > 0 && tx < blockDim.x-1 && ty > 0 && ty < blockDim.y-1) {

        float c = tile[ty * pitch + tx];
        float u = tile[(ty-1) * pitch + tx];
        float d = tile[(ty+1) * pitch + tx];
        float l = tile[ty * pitch + (tx-1)];
        float r = tile[ty * pitch + (tx+1)];
        _grid_new[y*N + x] = c + k*(u+d+l+r - 4.0f*c);
    }
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

    // Block/grid config (mantengo tu lógica de Y a partir de tpb)
    dim3 block_dim(TILE_X, CEIL_DIV(threads_per_block, TILE_X));
    dim3 grid_dim(CEIL_DIV(N, block_dim.x), CEIL_DIV(N, block_dim.y));

    // ---- Shared memory size (padding-only): by * (bx+1) floats ----
    size_t shm_bytes = (size_t)block_dim.y * (block_dim.x + 1) * sizeof(float);

    printf("Grid dim: %d, %d\n", grid_dim.x, grid_dim.y);
    printf("Block dim: %d, %d\n", block_dim.x, block_dim.y);
    printf("Shared memory size: %zu bytes\n", shm_bytes);
    mantener_fuentes_de_calor<<<1, 1>>>(d_grid, grid_size);
    cudaDeviceSynchronize();

    double ti_total = dwalltime();
    // Iteraciones de simulación
    for (int step = 0; step < steps; ++step) {
        actualizar_simulacion<<<grid_dim, block_dim, shm_bytes>>>(d_grid, d_grid_new, N, diffusion_rate);
        // Enforce sources on the freshly computed field
        mantener_fuentes_de_calor<<<1, 1>>>(d_grid_new, grid_size);
        // Now make the new field the current one
        tmp = d_grid; d_grid = d_grid_new; d_grid_new = tmp;
    }
    double tf_total = dwalltime();
    printf("Time taken: %f seconds\n", tf_total - ti_total);
    cudaDeviceSynchronize();
    cudaMemcpy(h_grid, d_grid, grid_size * grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    PRINT_ARRAY(h_grid, grid_size * grid_size);

    // Guardar para visualizar
    FILE* fp = fopen(output_path, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open output file '%s' for writing.\n", output_path);
        free(h_grid);
        cudaFree(d_grid);
        cudaFree(d_grid_new);
        return 1;
    }

    fprintf(fp, "%d\n", grid_size);
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
    free(h_grid);
    cudaFree(d_grid);
    cudaFree(d_grid_new);
    return 0;
}
