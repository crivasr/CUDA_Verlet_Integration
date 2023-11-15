#include "atoms.cuh"
#include "draw/draw.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__device__ float cudaDistance(Atom* a, Atom* b) {
    float distanceSq = powf(a->x - b->x, 2) + powf(a->y - b->y, 2);
    return sqrtf(distanceSq);
}

float distance(Atom* a, Atom* b) {
    float distanceSq = powf(a->x - b->x, 2) + powf(a->y - b->y, 2);
    return sqrtf(distanceSq);
}

__global__ void cudaUpdateVelocity(Atom* atoms, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    Atom* atom = &(atoms[idx]);

    atom->vx = (atom->x - atom->oldx) / (SUB_DELTA_T);
    atom->vy = (atom->y - atom->oldy) / (SUB_DELTA_T);
}

void updateVelocity(Atom* atoms, int size) {
    int blockSize, minGridSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaUpdateVelocity, 0, size);
    gridSize = (size + blockSize - 1) / blockSize;

    cudaUpdateVelocity<<<gridSize, blockSize>>>(atoms, size);
}

__global__ void cudaUpdateAtoms(Atom* atoms, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    Atom* atom = &(atoms[idx]);

    if (atom->fixed) return;
    atom->vy += GRAVITY * SUB_DELTA_T;

    float vy = atom->vy;
    float vx = atom->vx;

    atom->oldy = atom->y;
    atom->oldx = atom->x;

    atom->y += vy * SUB_DELTA_T;
    atom->x += vx * SUB_DELTA_T;
}

void updateAtoms(Atom* atoms, int size) {
    // https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum
                        // occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaUpdateAtoms, 0, size);
    gridSize = (size + blockSize - 1) / blockSize;

    cudaUpdateAtoms<<<gridSize, blockSize>>>(atoms, size);
}

__global__ void cudaConstrainAtoms(Atom* atoms, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > size) return;
    Atom* atom = &atoms[idx];

    float vy = (atom->y - atom->oldy) * FRICTION_CONSTANT;
    float vx = (atom->x - atom->oldx) * FRICTION_CONSTANT;

    int lowerLimit = 0 + ATOM_RADIUS;
    int upperLimit = WINDOW_HEIGHT - ATOM_RADIUS;

    if (atom->y < lowerLimit) {
        atom->y = lowerLimit;
        atom->oldy = atom->y + vy * BOUNCE_CONSTANT;
    } else
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
    int blockSize, minGridSize, gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaConstrainAtoms, 0, size);
    gridSize = (size + blockSize - 1) / blockSize;
    cudaConstrainAtoms<<<gridSize, blockSize>>>(atoms, size);
}

__global__ void cudaDrawAtoms(Pixel* image, Atom* atoms, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > size) return;
    Atom* atom = &atoms[idx];
    Point c = {atom->x, atom->y};

    Pixel color = WHITE_PIXEL;

    cudaDrawCercle(image, c, ATOM_RADIUS, color);
}

void drawAtoms(Pixel* image, Atom* atoms, int size) {
    int blockSize, minGridSize, gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaConstrainAtoms, 0, size);
    gridSize = (size + blockSize - 1) / blockSize;

    Pixel* cudaImage = nullptr;

    cudaMalloc(&cudaImage, sizeof(Pixel) * WINDOW_HEIGHT * WINDOW_WIDTH);
    cudaMemcpy(cudaImage, image, sizeof(Pixel) * WINDOW_HEIGHT * WINDOW_WIDTH, cudaMemcpyHostToDevice);
    
    cudaDrawAtoms<<<gridSize, blockSize>>>(cudaImage, atoms, size);

    cudaMemcpy(image, cudaImage, sizeof(Pixel) * WINDOW_HEIGHT * WINDOW_WIDTH, cudaMemcpyDeviceToHost);

    cudaFree(cudaImage);
}

__global__ void cudaPreSolve(Atom* atoms, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > size) return;
    Atom* atom = &atoms[idx];

    atom->dx = 0;
    atom->dy = 0;
}

void preSolve(Atom* atoms, int size) {
    int blockSize, minGridSize, gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaConstrainAtoms, 0, size);
    gridSize = (size + blockSize - 1) / blockSize;

    cudaPreSolve<<<gridSize, blockSize>>>(atoms, size);
}

__global__ void cudaPostSolve(Atom* atoms, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > size) return;
    Atom* atom = &atoms[idx];

    atom->x += 0.25 * atom->dx;
    atom->y += 0.25 * atom->dy;
}

void postSolve(Atom* atoms, int size) {
    int blockSize, minGridSize, gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaConstrainAtoms, 0, size);
    gridSize = (size + blockSize - 1) / blockSize;

    cudaPostSolve<<<gridSize, blockSize>>>(atoms, size);
}