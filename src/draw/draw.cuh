#ifndef _CUDA_DRAW_CUH
#define _CUDA_DRAW_CUH

#include "defines.h"
#include "gui/gui.h"

typedef struct point {
    float x;
    float y;
} Point;

void drawCercle(Pixel* image, Point c, int radius, Pixel color);
void drawSquare(Pixel* image, Point c, int width, Pixel color);
void drawLine(Pixel* image, Point s, Point d, Pixel color);

__device__ void cudaDrawCercle(Pixel* image, Point c, int radius, Pixel color);
__device__ void cudaDrawSquare(Pixel* image, Point c, int width, Pixel color);
__device__ void cudaDrawLine(Pixel* image, Point s, Point d, Pixel color);

#endif