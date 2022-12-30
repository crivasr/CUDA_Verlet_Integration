#ifndef _CUDA_ATOMS_H
#define _CUDA_ATOMS_H

#include "../defines.h"

float distance(Atom* a, Atom* b);
void updateAtoms(Atom* atoms, int size);
void constrainAtoms(Atom* atoms, int size);
void drawAtoms(Pixel* image, Atom* atoms, int size);

#endif