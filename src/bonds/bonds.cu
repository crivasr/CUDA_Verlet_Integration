#include "../atoms/atoms.cuh"
#include "../atoms/atoms.h"
#include "../draw/draw.cuh"
#include "bonds.cuh"
#include "bonds.h"

__global__ void cudaUpdateBonds(Bond* bonds, Atom* atoms, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;
    Bond* bond = &bonds[idx];

    if (bond->broken) return;
    Atom* a = &atoms[bond->idxA];
    Atom* b = &atoms[bond->idxB];

    float aX = a->x;
    float aY = a->y;

    float bX = b->x;
    float bY = b->y;

    float dx = aX - bX;
    float dy = aY - bY;

    float dst = cudaDistance(a, b);

    if (dst > BOND_BROKEN_DISTANCE) {
        bond->broken = true;
        return;
    };

    float difference = bond->length - dst;
    float percent = difference / dst / 2;

    float offsetX = dx * percent;
    float offsetY = dy * percent;

    if (!a->fixed) {
        a->x = aX + offsetX;
        a->y = aY + offsetY;
    }
    if (!b->fixed) {
        b->x = bX - offsetX;
        b->y = bY - offsetY;
    }
}

void updateBonds(Bond* bonds, Atom* atoms, int sizeBonds, int sizeAtoms) {
#pragma omp parallel for
    for (int idx = 0; idx < sizeBonds; idx++) {
        Bond* bond = &bonds[idx];

        if (bond->broken) continue;
        Atom* a = &atoms[bond->idxA];
        Atom* b = &atoms[bond->idxB];

        float aX = a->x;
        float aY = a->y;

        float bX = b->x;
        float bY = b->y;

        float dx = aX - bX;
        float dy = aY - bY;

        float dst = distance(a, b);

        if (dst > BOND_BROKEN_DISTANCE) {
            bond->broken = true;
            continue;
        };

        float difference = bond->length - dst;
        float percent = difference / dst / 2;

        float offsetX = dx * percent;
        float offsetY = dy * percent;

        if (!a->fixed) {
            a->x = aX + offsetX;
            a->y = aY + offsetY;
        }
        if (!b->fixed) {
            b->x = bX - offsetX;
            b->y = bY - offsetY;
        }
    }
    return;
    int blockSize = 1;
    int minGridSize;
    int gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaUpdateBonds, 0, sizeBonds);
    gridSize = (sizeBonds + blockSize - 1) / blockSize;

    Bond* cudaBonds = nullptr;
    Atom* cudaAtoms = nullptr;

    cudaMalloc(&cudaBonds, sizeof(Bond) * sizeBonds);
    cudaMalloc(&cudaAtoms, sizeof(Atom) * sizeAtoms);

    cudaMemcpy(cudaBonds, bonds, sizeof(Bond) * sizeBonds, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaAtoms, atoms, sizeof(atom) * sizeAtoms, cudaMemcpyHostToDevice);

    cudaUpdateBonds<<<gridSize, blockSize>>>(cudaBonds, cudaAtoms, sizeBonds);

    cudaMemcpy(bonds, cudaBonds, sizeof(Bond) * sizeBonds, cudaMemcpyDeviceToHost);
    cudaMemcpy(atoms, cudaAtoms, sizeof(Atom) * sizeAtoms, cudaMemcpyDeviceToHost);

    cudaFree(cudaBonds);
    cudaFree(cudaAtoms);
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

void drawBonds(Pixel* image, Bond* bonds, Atom* atoms, int sizeBonds, int sizeAtoms) {
    int blockSize = 1;
    int minGridSize;
    int gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaDrawBonds, 0, sizeBonds);
    gridSize = (sizeBonds + blockSize - 1) / blockSize;

    Pixel* cudaImage = nullptr;
    Bond* cudaBonds = nullptr;
    Atom* cudaAtoms = nullptr;

    cudaMalloc(&cudaImage, sizeof(Pixel) * WINDOW_HEIGHT * WINDOW_WIDTH);
    cudaMalloc(&cudaBonds, sizeof(Bond) * sizeBonds);
    cudaMalloc(&cudaAtoms, sizeof(Atom) * sizeAtoms);

    cudaMemcpy(cudaImage, image, sizeof(Pixel) * WINDOW_HEIGHT * WINDOW_WIDTH, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaBonds, bonds, sizeof(Bond) * sizeBonds, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaAtoms, atoms, sizeof(atom) * sizeAtoms, cudaMemcpyHostToDevice);

    cudaDrawBonds<<<gridSize, blockSize>>>(cudaImage, cudaBonds, cudaAtoms, sizeBonds);

    cudaMemcpy(image, cudaImage, sizeof(Pixel) * WINDOW_HEIGHT * WINDOW_WIDTH, cudaMemcpyDeviceToHost);
    cudaMemcpy(bonds, cudaBonds, sizeof(Bond) * sizeBonds, cudaMemcpyDeviceToHost);
    cudaMemcpy(atoms, cudaAtoms, sizeof(Atom) * sizeAtoms, cudaMemcpyDeviceToHost);

    cudaFree(cudaImage);
    cudaFree(cudaBonds);
    cudaFree(cudaAtoms);
}

__device__ bool cudaIntersects(Point point1, Point point2, Point point3, Point point4) {
    float x1 = point1.x;
    float x2 = point2.x;
    float x3 = point3.x;
    float x4 = point4.x;

    float y1 = point1.y;
    float y2 = point2.y;
    float y3 = point3.y;
    float y4 = point4.y;

    float tDeno = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4);
    float tDivi = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    float t = tDeno / tDivi;

    float uDeno = (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2);
    float uDivi = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    float u = uDeno / uDivi;

    return t >= 0 && t <= 1 && u >= 0 && u <= 1;
}

__global__ void cudaCutBonds(Bond* bonds, Atom* atoms, int size, Point point3, Point point4) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;
    Bond* bond = &bonds[idx];

    Atom* a = &atoms[bond->idxA];
    Atom* b = &atoms[bond->idxB];

    Point point1 = {a->x, a->y};
    Point point2 = {b->x, b->y};

    if (cudaIntersects(point1, point2, point3, point4)) {
        bond->broken = true;
    }
}

void cutBonds(Bond* bonds, Atom* atoms, int sizeBonds, int sizeAtoms, Point point3, Point point4) {
    int blockSize = 1;
    int minGridSize;
    int gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaCutBonds, 0, sizeBonds);
    gridSize = (sizeBonds + blockSize - 1) / blockSize;

    Bond* cudaBonds = nullptr;
    Atom* cudaAtoms = nullptr;

    cudaMalloc(&cudaBonds, sizeof(Bond) * sizeBonds);
    cudaMalloc(&cudaAtoms, sizeof(Atom) * sizeAtoms);

    cudaMemcpy(cudaBonds, bonds, sizeof(Bond) * sizeBonds, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaAtoms, atoms, sizeof(Atom) * sizeAtoms, cudaMemcpyHostToDevice);

    cudaCutBonds<<<gridSize, blockSize>>>(cudaBonds, cudaAtoms, sizeBonds, point3, point4);

    cudaMemcpy(bonds, cudaBonds, sizeof(Bond) * sizeBonds, cudaMemcpyDeviceToHost);
    cudaMemcpy(atoms, cudaAtoms, sizeof(Atom) * sizeAtoms, cudaMemcpyDeviceToHost);

    cudaFree(cudaBonds);
    cudaFree(cudaAtoms);
}