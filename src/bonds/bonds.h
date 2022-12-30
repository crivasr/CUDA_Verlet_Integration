#ifndef _CUDA_BONDS_H
#define _CUDA_BONDS_H

#include "../defines.h"

void updateBonds(Bond* bonds, Atom* atoms, int sizeBonds, int sizeAtoms);
void drawBonds(Pixel* image, Bond* bonds, Atom* atoms, int sizeBonds, int sizeAtoms);
void cutBonds(Bond* bonds, Atom* atoms, int sizeBonds, int sizeAtoms, Point point3, Point point4);

#endif