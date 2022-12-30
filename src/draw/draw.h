#ifndef _C_DRAW_H
#define _C_DRAW_H

#include "../defines.h"

void drawCercle(Pixel* image, Point c, int radius, Pixel color);
void drawSquare(Pixel* image, Point c, int width, Pixel color);
void drawLine(Pixel* image, Point s, Point d, Pixel color);

#endif