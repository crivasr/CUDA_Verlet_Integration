#ifndef _DEFINES_H
#define _DEFINES_H

#include <cstdint>

#define DELTA_T ((float) 1.0 / (float) 60.0)
#define SUB_STEPS ((float) 30)
#define SUB_DELTA_T ((DELTA_T) / (SUB_STEPS))

#define GRAVITY ((float) -280.0)
#define BOUNCE_CONSTANT 0.9
#define FRICTION_CONSTANT 0.99

#define WINDOW_WIDTH 700
#define WINDOW_HEIGHT 700

#define BOND_UPDATES 1
#define BOND_BROKEN_DISTANCE 500
#define ATOM_RADIUS 1

#endif
