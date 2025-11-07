#include <GL/freeglut.h>
#include <stdio.h>
#include <stdlib.h>
#include "heat_simulation.h"

#define WIDTH 640
#define HEIGHT 480

float zoom = 1.0f;
float offsetX = 0.0f;
float offsetY = 0.0f;

unsigned int ms = 30; // milisegundos - equivale a Frames por Segundo -> FPSs= 1000/ (milisegundos por frame) -> 30ms son 33 FPS, 100ms son 10FPS

float clamp(float val, float min, float max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

void temperatura_a_color(float temp, float* r, float* g, float* b) {
    temp = clamp(temp, 0.0f, 100.0f);

    if (temp < 50.0f) {
        *r = 0.0f;
        *g = temp / 50.0f;
        *b = 1.0f - (*g);
    } else {
        *r = (temp - 50.0f) / 50.0f;
        *g = 1.0f - (*r);
        *b = 0.0f;
    }
}

void dibujarSuperficie(){
float r, g, b;
float cell_size = 1.0f; // Cada celda mide 1 unidad si se usa gluOrtho2D(0, grid_size, 0, grid_size)

    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            // Color de la celda (RGB normalizado a [0,1])
            temperatura_a_color(grid[y*grid_size + x], &r, &g, &b);
            glColor3f(r, g, b);

            // Dibujar el cuadrado de la celda
            glBegin(GL_QUADS);
                glVertex2f(y, x);
                glVertex2f(y + cell_size, x);
                glVertex2f(y + cell_size, x + cell_size);
                glVertex2f(y, x + cell_size);
            glEnd();
        }
    }
}

void display(void){
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    dibujarSuperficie();

    glutSwapBuffers();
}

void timer(int value) {
    update_simulation();
    glutPostRedisplay();
    glutTimerFunc(ms, timer, 0);
}

void updateProjection(int w, int h){
    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    float view_size = grid_size / zoom;

    gluOrtho2D(offsetX, offsetX + view_size, offsetY, offsetY + view_size);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void reshape(int w, int h){
    updateProjection(w,h);
}

void keyboardKeys(unsigned char key, int x, int y){
    if (key == 27) { //ESC
        destroy__grid();
        exit(0);
    }
}

void specialKeys(int key, int x, int y){
float move_step = grid_size * 0.05f / zoom;

    switch (key) {
        case GLUT_KEY_UP:    offsetY -= move_step; break;
        case GLUT_KEY_DOWN:  offsetY += move_step; break;
        case GLUT_KEY_LEFT:  offsetX += move_step; break;
        case GLUT_KEY_RIGHT: offsetX -= move_step; break;
    }

    updateProjection(WIDTH, HEIGHT);
    glutPostRedisplay();
}

void mouseWheel(int wheel, int direction, int x, int y){
// Tamaño visible antes del zoom
float view_size_old = grid_size / zoom;

    if (direction > 0)
        zoom *= 1.1f;   // acercar
    else
        zoom /= 1.1f;   // alejar

    if (zoom < 0.1f) zoom = 0.1f;

    // Tamaño visible despues del zoom
    float view_size_new = grid_size / zoom;

    // Centro actual antes del zoom
    float centerX = offsetX + view_size_old / 2.0f;
    float centerY = offsetY + view_size_old / 2.0f;

    // Modificar offset para mantener el centro en el mismo lugar
    offsetX = centerX - view_size_new / 2.0f;
    offsetY = centerY - view_size_new / 2.0f;

    updateProjection(WIDTH, HEIGHT);
    glutPostRedisplay();
}


int main(int argc, char* argv[]) {
int N = 400;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Simulación de Calor 2D (Modelo separado)");

    initialize_grid(N);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutSpecialFunc(specialKeys);
    glutKeyboardFunc(keyboardKeys);
    glutMouseWheelFunc(mouseWheel);

    glutTimerFunc(ms, timer, 0); //Llama a timer (el paso de simulacion) cada ms milisegundos - FPSs= 1000/ (milisegundos por frame)

    glPointSize(1.0f);

    glutMainLoop();

    destroy__grid();
    return 0;
}
