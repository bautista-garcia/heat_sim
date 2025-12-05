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
int N, threads_per_block;
float diffusion_rate = 0.25f;
int steps = 10000;
float tolerance = 0.00001f;
const char* output_path = "simulation_output.txt";


float *d_grid, *d_grid_new, *h_grid, *tmp; 
int grid_size;       

#ifdef PROFILE
double start_total, end_total;
double total_time_diffusion = 0.0, total_time_mantener = 0.0, total_time_reduce = 0.0;
double total_time_copy_d2d = 0.0, total_time_copy_d2h = 0.0;
double start_diffusion, end_diffusion, start_mantener, end_mantener, start_reduce, end_reduce;
double start_copy_d2d, end_copy_d2d, start_copy_d2h, end_copy_d2h;
int num_steps_executed = 0;
#endif

// Configuracion de difusion
dim3 block_diffusion;
dim3 grid_diffusion;
size_t shm_bytes_diffusion;

// Configuracion de mantener de fuentes de calor
dim3 block_mantener(1, 1);
dim3 grid_mantener(1, 1);
size_t shm_bytes_mantener = 0;

// Configuracion inicial de reduccion
dim3 block_reduce;
dim3 grid_reduce;
size_t shm_bytes_reduce;
int len_reduce = grid_size * grid_size;
float h_result_reduce, h_result_reduce_prev;

double dwalltime(){
    double sec;
    struct timeval tv;

    gettimeofday(&tv,NULL);
    sec = tv.tv_sec + tv.tv_usec/1000000.0;
    return sec;
}



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

    // Recalcular configuraciones de kernels con los valores correctos
    block_diffusion = dim3(TILE_X, CEIL_DIV(threads_per_block, TILE_X));
    grid_diffusion = dim3(CEIL_DIV(N, block_diffusion.x), CEIL_DIV(N, block_diffusion.y));
    shm_bytes_diffusion = (size_t)(block_diffusion.y + HALO_SM) * (block_diffusion.x + HALO_SM) * sizeof(float);
    
    block_reduce = dim3(threads_per_block, 1);
    grid_reduce = dim3(CEIL_DIV(N * N, threads_per_block * 2), 1);
    shm_bytes_reduce = threads_per_block * sizeof(float);

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
    #ifdef PROFILE
    start_diffusion = dwalltime();
    #endif
    difusion_kernel<<<grid_diffusion, block_diffusion, shm_bytes_diffusion>>>(d_grid, d_grid_new, grid_size, diffusion_rate);
    cudaDeviceSynchronize();
    #ifdef PROFILE
    end_diffusion = dwalltime();
    total_time_diffusion += (end_diffusion - start_diffusion);
    #endif
    // Recuperamos fuentes de calor
    #ifdef PROFILE
    start_mantener = dwalltime();
    #endif
    mantener_fuentes_de_calor<<<1, 1>>>(d_grid_new, grid_size);
    cudaDeviceSynchronize();
    #ifdef PROFILE
    end_mantener = dwalltime();
    total_time_mantener += (end_mantener - start_mantener);
    #endif

    // Reduccion (iterativa en device) + chequeo de convergencia en host
    #ifdef PROFILE
    start_copy_d2d = dwalltime();
    #endif
    cudaMemcpy(d_grid, d_grid_new, grid_size * grid_size * sizeof(float), cudaMemcpyDeviceToDevice);
    #ifdef PROFILE
    end_copy_d2d = dwalltime();
    total_time_copy_d2d += (end_copy_d2d - start_copy_d2d);
    #endif
    grid_reduce.x = CEIL_DIV(grid_size * grid_size, threads_per_block * 2);
    int len_reduce = grid_size * grid_size;
    
    while (grid_reduce.x > 1) {
        #ifdef PROFILE
        start_reduce = dwalltime();
        #endif
        reduction_kernel<<<grid_reduce, block_reduce, shm_bytes_reduce>>>(d_grid, d_grid, len_reduce);
        cudaDeviceSynchronize();
        #ifdef PROFILE
        end_reduce = dwalltime();
        total_time_reduce += (end_reduce - start_reduce);
        #endif
        len_reduce = grid_reduce.x;
        grid_reduce.x = CEIL_DIV(len_reduce, threads_per_block * 2);
    }
    #ifdef PROFILE
    start_reduce = dwalltime();
    #endif
    reduction_kernel<<<grid_reduce, block_reduce, shm_bytes_reduce>>>(d_grid, d_grid, len_reduce);
    cudaDeviceSynchronize();
    #ifdef PROFILE
    end_reduce = dwalltime();
    total_time_reduce += (end_reduce - start_reduce);
    #endif

    #ifdef PROFILE
    start_copy_d2h = dwalltime();
    #endif
    cudaMemcpy(&h_result_reduce, d_grid, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grid, d_grid_new, grid_size * grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    #ifdef PROFILE
    end_copy_d2h = dwalltime();
    total_time_copy_d2h += (end_copy_d2h - start_copy_d2h);
    #endif
    if ((fabs(h_result_reduce - h_result_reduce_prev) / (grid_size * grid_size)) < tolerance) {
        converged = true;
    }
    h_result_reduce_prev = h_result_reduce;

    // Swap de punteros
    float* tmp = d_grid; 
    d_grid = d_grid_new; 
    d_grid_new = tmp;

    return converged;
}



int main(int argc, char** argv){
    // Device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    PRINT_DEVICE_SUMMARY(prop);

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <N> <threads_per_block>\n", argv[0]);
        return 1;
    }

    // Parametrizacion 
    N = atoi(argv[1]); // Tamaño de la grilla
    threads_per_block = atoi(argv[2]);

    start_total = dwalltime();
    initialize_grid(N);
    bool converged = false;
    num_steps_executed = 0;
    for (int step = 0; step < steps; ++step) {
        converged = update_simulation();
        num_steps_executed++;
        if (converged) {
            break;
        }
    }

    cudaDeviceSynchronize();
    #ifdef PROFILE
    start_copy_d2h = dwalltime();
    #endif
    cudaMemcpy(h_grid, d_grid, grid_size * grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    #ifdef PROFILE
    end_copy_d2h = dwalltime();
    total_time_copy_d2h += (end_copy_d2h - start_copy_d2h);
    #endif

    end_total = dwalltime();
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

    #ifdef PROFILE
    printf("\n=== RESULTADOS DE PROFILING ===\n");
    printf("Configuracion: N = %d, threads_per_block = %d, epsilon = %f\n", N, threads_per_block, tolerance);
    printf("Tiempo total de ejecución: %.6f segundos\n", end_total - start_total);
    printf("Pasos ejecutados: %d\n", num_steps_executed);
    if (num_steps_executed > 0) {
        printf("Tiempo promedio por paso en kernel de difusión: %.8f segundos\n", total_time_diffusion / num_steps_executed);
        printf("Tiempo promedio por paso en kernel mantener fuentes: %.8f segundos\n", total_time_mantener / num_steps_executed);
        printf("Tiempo promedio por paso en kernel de reducción: %.8f segundos\n", total_time_reduce / num_steps_executed);
        printf("Tiempo promedio por paso en copias Device-to-Device: %.8f segundos\n", total_time_copy_d2d / num_steps_executed);
        printf("Tiempo promedio por paso en copias Device-to-Host: %.8f segundos\n", total_time_copy_d2h / num_steps_executed);
    }
    printf("================================\n");
    #endif

    destroy__grid();
    return 0;
}
