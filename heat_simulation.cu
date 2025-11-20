#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "cuda_utils.h"
#include "heat_simulation.h"
#define TILE_X 32
#define HALO_SM 2


// Parametros de simulacion
int N = 64;
int threads_per_block = 128;
float diffusion_rate = 0.25f;
int steps = 10000;
const char* output_path = "simulation_output.txt";


float *d_grid, *d_grid_new, *h_grid, *tmp; 
int grid_size;       

// Configuracion de difusion
dim3 block_diffusion(TILE_X, CEIL_DIV(threads_per_block, TILE_X));
dim3 grid_diffusion(CEIL_DIV(N, block_diffusion.x), CEIL_DIV(N, block_diffusion.y));
size_t shm_bytes_diffusion = (size_t)(block_diffusion.y + HALO_SM) * (block_diffusion.x + HALO_SM) * sizeof(float);

// Configuracion de mantener de fuentes de calor
dim3 block_mantener(1, 1);
dim3 grid_mantener(1, 1);
size_t shm_bytes_mantener = 0;

// Configuracion inicial de reduccion
dim3 block_reduce(threads_per_block, 1);
dim3 grid_reduce(CEIL_DIV(N * N, threads_per_block * 2), 1);
size_t shm_bytes_reduce = threads_per_block * sizeof(float);
int len_reduce = grid_size * grid_size;
float h_result_reduce, h_result_reduce_prev;


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

__global__ void difusion_kernel(const float* __restrict__ _grid, float* __restrict__ _grid_new, int grid_size, float k) {
    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * grid_size + x;

    if (x >= grid_size || y >= grid_size) return;

    // Memoria compartida 
    extern __shared__ float smem[];
    const int stride = blockDim.x + HALO_SM;
    const int lx = threadIdx.x + 1;
    const int ly = threadIdx.y + 1;

    smem[ly * stride + lx] = _grid[i]; 
    // Top Halo
    if ((threadIdx.y == 0) && (y > 0)) { // Top Halo
        smem[0 * stride + lx] = _grid[i - grid_size];
    }
    if ((threadIdx.y == blockDim.y - 1) && (y < grid_size - 1)) { // Bottom Halo
        smem[(blockDim.y + 1) * stride + lx] = _grid[i + grid_size];
    }
    
    if ((threadIdx.x == 0) && (x > 0)) { // Left Halo
        smem[ly * stride + 0] = _grid[i - 1];
    }
    if ((threadIdx.x == blockDim.x - 1) && (x < grid_size - 1)) { // Right Halo
        smem[ly * stride + (blockDim.x + 1)] = _grid[i + 1];
    }
    
    __syncthreads();
    
    // No se procesan los bordes
    if (x > 0 && x < grid_size - 1 && y > 0 && y < grid_size - 1) {
        float c = smem[ly * stride + lx];
        float u = smem[(ly - 1) * stride + lx];
        float d = smem[(ly + 1) * stride + lx];
        float l = smem[ly * stride + (lx - 1)];
        float r = smem[ly * stride + (lx + 1)];
        
        _grid_new[i] = c + k * (u + d + l + r - 4.0f * c);
    } else { // Para bordes copiamos valor anterior
        if (x < grid_size && y < grid_size) {
             _grid_new[i] = _grid[i];
        }
    }
}

__global__ void reduction_kernel(const float* __restrict__ input, float* __restrict__ output, int n) {
    extern __shared__ float sdata[];
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


__host__ void initialize_grid(int N) {
    grid_size = N;
    // Grilla NxN + padding global          
    h_grid = (float*)malloc(sizeof(float) * N * N);
    h_result_reduce = 0.0f;
    h_result_reduce_prev = -1.0f;

    size_t bytes = sizeof(float) * N * N;
    cudaMalloc(&d_grid, bytes);
    cudaMalloc(&d_grid_new, bytes);

    cudaMemset(d_grid, 0, bytes);
    cudaMemset(d_grid_new, 0, bytes);
    mantener_fuentes_de_calor<<<1, 1>>>(d_grid, grid_size);
}



__host__ void destroy__grid(){
    free(h_grid);
    h_grid = NULL;
    cudaFree(d_grid);
    d_grid = NULL;
    cudaFree(d_grid_new);
    d_grid_new = NULL;
    grid_size = 0;
}

__host__ bool update_simulation() {
    bool converged = false;
    // Paso de difusion
    difusion_kernel<<<grid_diffusion, block_diffusion, shm_bytes_diffusion>>>(d_grid, d_grid_new, grid_size, diffusion_rate);
    cudaDeviceSynchronize();
    // Recuperamos fuentes de calor
    mantener_fuentes_de_calor<<<1, 1>>>(d_grid_new, grid_size);

    // Reduccion (iterativa en device) + chequeo de convergencia en host
    cudaMemcpy(d_grid, d_grid_new, grid_size * grid_size * sizeof(float), cudaMemcpyDeviceToDevice);
    grid_reduce.x = CEIL_DIV(N * N, threads_per_block * 2);
    int len_reduce = grid_size * grid_size;
    
    while (grid_reduce.x > 1) {
        reduction_kernel<<<grid_reduce, block_reduce, shm_bytes_reduce>>>(d_grid, d_grid, len_reduce);
        cudaDeviceSynchronize();
        len_reduce = grid_reduce.x;
        grid_reduce.x = CEIL_DIV(len_reduce, threads_per_block * 2);
    }
    reduction_kernel<<<grid_reduce, block_reduce, shm_bytes_reduce>>>(d_grid, d_grid, len_reduce);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_result_reduce, d_grid, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grid, d_grid_new, grid_size * grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (fabs(h_result_reduce - h_result_reduce_prev) < 0.0001f) {
        converged = true;
    }
    h_result_reduce_prev = h_result_reduce;

    // Swap de punteros
    float* tmp = d_grid; 
    d_grid = d_grid_new; 
    d_grid_new = tmp;

    return converged;
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

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <grid_size> <threads_per_block> <diffusion_rate> <steps> [output_path]\n", argv[0]);
        return 1;
    }

    // Parametrizacion 
    N = atoi(argv[1]); // Tamaño de la grilla
    threads_per_block = atoi(argv[2]);

    initialize_grid(N);
    bool converged = false;
    for (int step = 0; step < steps; ++step) {
        converged = update_simulation();
        if (converged) {
            printf("Simulation converged at step %d\n", step);
            break;
        }
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_grid, d_grid, grid_size * grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    // --------------------------------------------------------------
    // VISUALIZACION CON PYTHON 
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
    // --------------------------------------------------------------

    destroy__grid();
    return 0;
}
