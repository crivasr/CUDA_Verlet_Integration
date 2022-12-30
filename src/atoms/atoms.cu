#include "../draw/draw.cuh"
#include "atoms.cuh"
#include "atoms.h"

__device__ float cudaDistance(Atom* a, Atom* b) {
    float distanceSq = powf(a->x - b->x, 2) + powf(a->y - b->y, 2);
    return sqrtf(distanceSq);
}

__global__ void cudaUpdateAtoms(Atom* atoms, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    Atom* atom = &(atoms[idx]);

    if (atom->fixed) return;
    float vy = (atom->y - atom->oldy) * FRICTION_CONSTANT;
    float vx = (atom->x - atom->oldx) * FRICTION_CONSTANT;

    atom->oldy = atom->y;
    atom->oldx = atom->x;

    atom->y += vy + GRAVITY_CONSTANT;
    atom->x += vx;
}

void updateAtoms(Atom* atoms, int size) {
    // //
    // https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
    int blockSize = 1;  // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum
                        // occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaUpdateAtoms, 0, size);
    gridSize = (size + blockSize - 1) / blockSize;

    Atom* cudaAtoms = 0;
    cudaMalloc(&cudaAtoms, sizeof(Atom) * size);
    cudaMemcpy(cudaAtoms, atoms, sizeof(Atom) * size, cudaMemcpyHostToDevice);
    cudaUpdateAtoms<<<gridSize, blockSize>>>(cudaAtoms, size);
    cudaMemcpy(atoms, cudaAtoms, sizeof(Atom) * size, cudaMemcpyDeviceToHost);
    cudaFree(cudaAtoms);
}

__global__ void cudaConstrainAtoms(Atom* atoms, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > size) return;
    Atom* atom = &atoms[idx];

    float vy = (atom->y - atom->oldy) * FRICTION_CONSTANT;
    float vx = (atom->x - atom->oldx) * FRICTION_CONSTANT;

    int lowerLimit = 0 + ATOM_RADIUS;
    int upperLimit = WINDOW_HEIGHT - ATOM_RADIUS;

    // if (atom->y < lowerLimit) {
    //     atom->y = lowerLimit;
    //     atom->oldy = atom->y + vy * BOUNCE_CONSTANT;
    // } else
    if (atom->y > upperLimit) {
        atom->y = upperLimit;
        atom->oldy = atom->y + vy * BOUNCE_CONSTANT;
    }

    if (atom->x < lowerLimit) {
        atom->x = lowerLimit;
        atom->oldx = atom->x + vx * BOUNCE_CONSTANT;
    } else if (atom->x > WINDOW_WIDTH - ATOM_RADIUS) {
        atom->x = upperLimit;
        atom->oldx = atom->x + vx * BOUNCE_CONSTANT;
    }
}

void constrainAtoms(Atom* atoms, int size) {
    int blockSize = 1;
    int minGridSize;
    int gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaConstrainAtoms, 0, size);
    gridSize = (size + blockSize - 1) / blockSize;

    Atom* cudaAtoms = nullptr;
    cudaMalloc(&cudaAtoms, sizeof(Atom) * size);
    cudaMemcpy(cudaAtoms, atoms, sizeof(Atom) * size, cudaMemcpyHostToDevice);
    cudaConstrainAtoms<<<gridSize, blockSize>>>(cudaAtoms, size);
    cudaMemcpy(atoms, cudaAtoms, sizeof(Atom) * size, cudaMemcpyDeviceToHost);
    cudaFree(cudaAtoms);
}

__global__ void cudaDrawAtoms(Pixel* image, Atom* atoms, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > size) return;
    Atom* atom = &atoms[idx];
    Point c = {atom->x, atom->y};

    Pixel color = WHITE_PIXEL;

    cudaDrawSquare(image, c, ATOM_RADIUS, color);
}

void drawAtoms(Pixel* image, Atom* atoms, int size) {
    int blockSize = 1;
    int minGridSize;
    int gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaDrawAtoms, 0, size);
    gridSize = (size + blockSize - 1) / blockSize;

    Pixel* cudaImage = nullptr;
    Atom* cudaAtoms = nullptr;

    cudaMalloc(&cudaImage, sizeof(Pixel) * WINDOW_HEIGHT * WINDOW_WIDTH);
    cudaMemcpy(cudaImage, image, sizeof(Pixel) * WINDOW_HEIGHT * WINDOW_WIDTH, cudaMemcpyHostToDevice);

    cudaMalloc(&cudaAtoms, sizeof(Atom) * size);
    cudaMemcpy(cudaAtoms, atoms, sizeof(Atom) * size, cudaMemcpyHostToDevice);

    cudaDrawAtoms<<<gridSize, blockSize>>>(cudaImage, cudaAtoms, size);

    cudaMemcpy(image, cudaImage, sizeof(Pixel) * WINDOW_HEIGHT * WINDOW_WIDTH, cudaMemcpyDeviceToHost);
    cudaMemcpy(atoms, cudaAtoms, sizeof(Atom) * size, cudaMemcpyDeviceToHost);

    cudaFree(cudaImage);
    cudaFree(cudaAtoms);
}