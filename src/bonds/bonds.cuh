#ifndef _CUDA_BONDS_H
#define _CUDA_BONDS_H

#include "defines.h"
#include "atoms/atoms.cuh"
#include "draw/draw.cuh"

typedef struct bond {
    int idxA;
    int idxB;
    float length;
    bool broken;

    bond(int _idxA, int _idxB, float _length): idxA(_idxA), idxB(_idxB), length(_length), broken(false) {}
} Bond;

void updateBonds(Bond* bonds, Atom* atoms, int sizeBonds);
void drawBonds(Pixel* image, Bond* bonds, Atom* atoms, int sizeBonds);

__global__ void cudaUpdateBonds(Bond* bonds, Atom* atoms, int size);
__global__ void cudaDrawBonds(Pixel* image, Bond* bonds, Atom* atoms, int size);

#endif