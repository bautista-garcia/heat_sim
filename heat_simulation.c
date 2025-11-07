#include <stdlib.h>
#include "heat_simulation.h"

float* new_grid;

float diffusion_rate = 0.25f;

float *grid = NULL;
int grid_size = 0;

void mantener_fuentes_de_calor(float* _grid){
    int cx = grid_size / 2;
    int cy = grid_size / 2;

    _grid[cy*grid_size+cx] = 100.0f;

    int offset = 20;
    _grid[(cy+offset)*grid_size + (cx+offset)] = 100.0f;
    _grid[(cy+offset)*grid_size + (cx-offset)] = 100.0f;
    _grid[(cy-offset)*grid_size + (cx+offset)] = 100.0f;
    _grid[(cy-offset)*grid_size + (cx-offset)] = 100.0f;
}

void initialize_grid(int N) {

    grid_size = N;
    grid = (float*)malloc(sizeof(float)*grid_size*grid_size);
    for (int i = 0; i < grid_size*grid_size; i++) {
        grid[i] = 0.0f;
    }

    new_grid = (float*)malloc(sizeof(float)*grid_size*grid_size);
    mantener_fuentes_de_calor(grid);
}

void update_simulation() {
float sum;
    for (int y = 1; y < grid_size - 1; y++) {
        for (int x = 1; x < grid_size - 1; x++) {
            sum = grid[(y - 1)*grid_size + x] + grid[(y + 1)*grid_size + x] +
                        grid[y*grid_size + (x - 1)] + grid[y*grid_size + (x + 1)];
            new_grid[y*grid_size + x] = grid[y*grid_size + x] + diffusion_rate * (sum - 4 * grid[y*grid_size + x]);
        }
    }

    mantener_fuentes_de_calor(new_grid);

    for (int y = 1; y < grid_size - 1; y++) {
        for (int x = 1; x < grid_size - 1; x++) {
            grid[y*grid_size + x] = new_grid[y*grid_size + x];
        }
    }
}

void destroy__grid(){
    free(grid);
    grid = NULL;
    grid_size = 0;
    free(new_grid);
}
