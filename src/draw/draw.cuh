#ifndef _CUDA_DRAW_CUH
#define _CUDA_DRAW_CUH

#include "../defines.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ void cudaDrawCercle(Pixel* image, Point c, int radius, Pixel color);
__device__ void cudaDrawSquare(Pixel* image, Point c, int width, Pixel color);
__device__ void cudaDrawLine(Pixel* image, Point s, Point d, Pixel color);

#endif