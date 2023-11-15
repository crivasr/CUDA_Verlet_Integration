#ifndef _CUDA_ATOMS_CUH
#define _CUDA_ATOMS_CUH

#include "defines.h"
#include "gui/gui.h"
#include "draw/draw.cuh"

typedef struct atom {
    float x;
    float y;
    float oldx;
    float oldy;
    float dx;
    float dy;
    float vx;
    float vy;
    bool fixed;

    atom(float _x, float _y, bool _fixed = false) :
        x(_x), y(_y), oldx(_x), oldy(_y),
        dx(0), dy(0), vx(0), vy(0), fixed(_fixed) {}
} Atom;

float distance(Atom* a, Atom* b);
void updateAtoms(Atom* atoms, int size);
void updateVelocity(Atom* atoms, int size);
void constrainAtoms(Atom* atoms, int size);
void drawAtoms(Pixel* image, Atom* atoms, int size);
void preSolve(Atom* atoms, int size);
void postSolve(Atom* atoms, int size);

__device__ float cudaDistance(Atom* a, Atom* b);
__global__ void cudaUpdateAtoms(Atom* atoms, int size);
__global__ void cudaUpdateVelocity(Atom* atoms, int size);
__global__ void cudaConstrainAtoms(Atom* atoms, int size);
__global__ void cudaDrawAtoms(Pixel* image, Atom* atoms, int size);
__global__ void cudaPreSolve(Atom* atoms, int size);
__global__ void cudaPostSolve(Atom* atoms, int size);

#endif