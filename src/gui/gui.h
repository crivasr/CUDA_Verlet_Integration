#ifndef _GUI_H
#define _GUI_H

#include "GLFW/glfw3.h"
#include "defines.h"

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


void clearImage(Pixel* image);
GLFWwindow* initWindow();
GLuint* initTexture();
Pixel* initImage();
void drawImage(Pixel* image, GLuint* texture);

#endif