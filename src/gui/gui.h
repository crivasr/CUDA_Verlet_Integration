#ifndef _GUI_H
#define _GUI_H

#include "../../lib/glfw/include/GLFW/glfw3.h"
#include "../defines.h"

void clearImage(Pixel* image);
GLFWwindow* initWindow();
GLuint* initTexture();
Pixel* initImage();
void drawImage(Pixel* image, GLuint* texture);

#endif