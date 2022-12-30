#include "atoms.h"

#include <math.h>

float distance(Atom* a, Atom* b) {
    float distanceSq = powf(a->x - b->x, 2) + powf(a->y - b->y, 2);
    return sqrtf(distanceSq);
}