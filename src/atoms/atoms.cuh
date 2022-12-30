#ifndef _CUDA_ATOMS_CUH
#define _CUDA_ATOMS_CUH

#include "../defines.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ float cudaDistance(Atom* a, Atom* b);
__global__ void cudaUpdateAtoms(Atom* atoms, int size);
__global__ void cudaConstrainAtoms(Atom* atoms, int size);

#endif