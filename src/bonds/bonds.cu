#include "bonds.cuh"
#include "atoms/atoms.cuh"
#include "draw/draw.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void cudaUpdateBonds(Bond* bonds, Atom* atoms, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;
    Bond* bond = &bonds[idx];

    if (bond->broken) return;
    Atom* a = &atoms[bond->idxA];
    Atom* b = &atoms[bond->idxB];

    
    float dst = cudaDistance(a, b);
    
    if (dst != 0) {
        if (dst > BOND_BROKEN_DISTANCE) {
            bond->broken = true;
            return;
        };
        
        float aX = a->x;
        float aY = a->y;
        
        float bX = b->x;
        float bY = b->y;
        
        float length = bond->length;
        if (!a->fixed) {
            atomicAdd(&a->dx, 0.2 * (length - dst) * (aX - bX) / dst);
            atomicAdd(&a->dy, 0.2 * (length - dst) * (aY - bY) / dst);
        }
        
        if (!b->fixed) {
            atomicAdd(&b->dx, - 0.2 * (length - dst) * (aX - bX) / dst);
            atomicAdd(&b->dy, - 0.2 * (length - dst) * (aY - bY) / dst);
        }
    }
}

void updateBonds(Bond* bonds, Atom* atoms, int sizeBonds) {
    int blockSize, minGridSize, gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaDrawBonds, 0, sizeBonds);
    gridSize = (sizeBonds + blockSize - 1) / blockSize;

    for (int _ = 0; _ < BOND_UPDATES; _++) {
        cudaUpdateBonds<<<gridSize, blockSize>>>(bonds, atoms, sizeBonds);
    }
}

__global__ void cudaDrawBonds(Pixel* image, Bond* bonds, Atom* atoms, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    Bond* bond = &bonds[idx];
    if (bond->broken) return;

    Atom* a = &atoms[bond->idxA];
    Atom* b = &atoms[bond->idxB];

    Point s = {a->x, a->y};
    Point d = {b->x, b->y};

    Pixel color = RED_PIXEL;

    cudaDrawLine(image, s, d, color);
}

void drawBonds(Pixel* image, Bond* bonds, Atom* atoms, int sizeBonds) {
    int blockSize, minGridSize, gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaDrawBonds, 0, sizeBonds);
    gridSize = (sizeBonds + blockSize - 1) / blockSize;

    Pixel* cudaImage;
    cudaMalloc(&cudaImage, sizeof(Pixel) * WINDOW_HEIGHT * WINDOW_WIDTH);
    cudaMemcpy(cudaImage, image, sizeof(Pixel) * WINDOW_HEIGHT * WINDOW_WIDTH, cudaMemcpyHostToDevice);

    cudaDrawBonds<<<gridSize, blockSize>>>(cudaImage, bonds, atoms, sizeBonds);

    cudaMemcpy(image, cudaImage, sizeof(Pixel) * WINDOW_HEIGHT * WINDOW_WIDTH, cudaMemcpyDeviceToHost);
    cudaFree(cudaImage);
}
