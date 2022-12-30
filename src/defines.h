#ifndef _DEFINES_H
#define _DEFINES_H

#include <cstdint>

#define GRAVITY_CONSTANT -0.3
#define BOUNCE_CONSTANT 0.9
#define FRICTION_CONSTANT 0.99

#define WINDOW_WIDTH 700
#define WINDOW_HEIGHT 700

#define BOND_UPDATES 10
#define BOND_BROKEN_DISTANCE 50
#define ATOM_RADIUS 1

typedef struct point {
    float x;
    float y;
} Point;

typedef struct atom {
    float x;
    float y;
    float oldx;
    float oldy;
    bool fixed;
} Atom;

typedef struct bond {
    int idxA;
    int idxB;
    float length;
    bool broken;
} Bond;

typedef struct pixel {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} Pixel;

#define RED_PIXEL \
    {             \
        255,      \
        0,        \
        0,        \
    };

#define GREEN_PIXEL \
    {               \
        0,          \
        255,        \
        0,          \
    };

#define BLUE_PIXEL \
    {              \
        0,         \
        0,         \
        255,       \
    };

#define WHITE_PIXEL \
    {               \
        255,        \
        255,        \
        255,        \
    };

#define BLACK_PIXEL \
    {               \
        0,          \
        0,          \
        0,          \
    };

#define YELLOW_PIXEL \
    {                \
        255,         \
        255,         \
        0,           \
    };

#define CYAN_PIXEL \
    {              \
        0,         \
        255,       \
        255,       \
    };

#define MAGENTA_PIXEL \
    {                 \
        255,          \
        0,            \
        255,          \
    };

#endif
