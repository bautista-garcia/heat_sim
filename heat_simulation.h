#ifndef HEAT_SIMULATION_H
#define HEAT_SIMULATION_H

extern float* grid;
extern int grid_size;

void initialize_grid(int N);
void update_simulation();
void mantener_fuentes_de_calor(float* _grid);
void destroy__grid();

#endif

