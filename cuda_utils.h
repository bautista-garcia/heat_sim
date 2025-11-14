#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>
#include <cuda_runtime.h>

#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))


#define PRINT_BLOCK_GRID(block, grid)                                      \
    do {                                                                   \
        printf("Blocks(x): %d, Blocks(y): %d, Blocks(z): %d, "             \
               "Grids(x): %d, Grids(y): %d, Grids(z): %d\n",               \
               (block).x, (block).y, (block).z,                            \
               (grid).x, (grid).y, (grid).z);                              \
    } while (0)


#define PRINT_DEVICE_PROP(prop)                                            \
    do {                                                                   \
        printf("Threads per block: %d, Max Grids(x): %d, Max Grids(y): %d, Max Grids(z): %d\n", \
               (prop).maxThreadsPerBlock,                                  \
               (prop).maxGridSize[0], (prop).maxGridSize[1], (prop).maxGridSize[2]); \
    } while (0)


#define PRINT_DEVICE_NAME(prop)                                            \
    do {                                                                   \
        printf("Device: %s\n", (prop).name);                               \
    } while (0)

#define PRINT_ARRAY(arr, n)                    \
do {                                           \
    for (int _i = 0; _i < (n); _i++) {         \
        printf("%.1f ", (arr)[_i]);           \
    }                                          \
    printf("\n");                            \
} while (0)

#define PRINT_DEVICE_SUMMARY(prop)                                                                     \
    do {                                                                                              \
        printf("======================= DEVICE SUMMARY =======================\n");                                                          \
        printf("Device: %s\n", (prop).name);                                                          \
        printf("Compute capability: %d.%d\n", (prop).major, (prop).minor);                            \
        printf("Shared mem/block: %zu\n", (size_t)(prop).sharedMemPerBlock);                          \
        printf("Shared mem/SM: %zu\n", (size_t)(prop).sharedMemPerMultiprocessor);                    \
        printf("SM count: %d\n", (prop).multiProcessorCount);                                         \
        printf("Registers/block: %d\n", (prop).regsPerBlock);                                         \
        printf("Registers/SM: %d\n", (prop).regsPerMultiprocessor);                                   \
        printf("==============================================================\n");                                   \
    } while (0)



#endif // CUDA_UTILS_H