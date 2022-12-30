#ifndef _CUDA_BONDS_CUH
#define _CUDA_BONDS_CUH

#include "../defines.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void cudaUpdateBonds(Bond* bonds, Atom* atoms, int size);
__global__ void cudaDrawBonds(Pixel* image, Bond* bonds, Atom* atoms, int size);

#endif