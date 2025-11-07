#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "cuda_utils.h"

// 2 grids para evitar conflictos de memoria en DEVICE y una en HOST para resultado final
float *d_grid, *d_grid_new, *h_grid; 
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

    int offset = 4;

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

__global__ void actualizar_simulacion(float* _grid, float* _grid_new, int grid_size, float diffusion_rate){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= grid_size || y >= grid_size) {
        return;
    }

    int idx = y * grid_size + x;

    if (y > 0 && y < grid_size - 1 && x > 0 && x < grid_size - 1) {
        _grid_new[idx] = _grid[idx] + diffusion_rate * (_grid[idx - 1] + _grid[idx + 1] + _grid[idx - grid_size] + _grid[idx + grid_size] - 4 * _grid[idx]);
    }

    _grid[idx] = _grid_new[idx];
}

void update_simulation(float* _grid, float* _grid_new, int grid_size, float diffusion_rate, int threads_per_block, dim3 grid_dim, dim3 block_dim){
    actualizar_simulacion<<<grid_dim, block_dim>>>(_grid, _grid_new, grid_size, diffusion_rate);
    cudaDeviceSynchronize();
    mantener_fuentes_de_calor<<<1, 1>>>(_grid, grid_size);
    cudaDeviceSynchronize();
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
    // double ti_transferencia, tf_transferencia;
    // double ti_total, tf_total;
    // double ti_kernel, tf_kernel;
    //Declaración de variables de CPU y GPU
    initialize_grid(N);
    dim3 block_dim(sqrt(threads_per_block), sqrt(threads_per_block));
    dim3 grid_dim(CEIL_DIV(N, block_dim.x), CEIL_DIV(N, block_dim.y));
    mantener_fuentes_de_calor<<<1, 1>>>(d_grid, grid_size);
    cudaDeviceSynchronize();
    double ti_total = dwalltime();
    for (int i = 0; i < steps; i++) {
        update_simulation(d_grid, d_grid_new, grid_size, diffusion_rate, threads_per_block, grid_dim, block_dim);
    }
    double tf_total = dwalltime();
    printf("Time taken: %f seconds\n", tf_total - ti_total);
    cudaDeviceSynchronize();
    cudaMemcpy(h_grid, d_grid, grid_size * grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    PRINT_ARRAY(h_grid, grid_size * grid_size);

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