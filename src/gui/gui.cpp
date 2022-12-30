#include "gui.h"

#include <cstdlib>
#include <iostream>

void clearImage(Pixel* image) {
    for (int x = 0; x < WINDOW_WIDTH; x++) {
        for (int y = 0; y < WINDOW_HEIGHT; y++) {
            Pixel* pixel = &image[x + y * WINDOW_HEIGHT];
            *pixel = BLACK_PIXEL;
        }
    }
}

GLFWwindow* initWindow() {
    if (!glfwInit()) {
        std::cout << "Couldn't init GLFW" << std::endl;
        exit(1);
    }

    GLFWwindow* window =
        glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Simulator", nullptr, nullptr);

    if (!window) {
        std::cout << "Couldn't open window" << std::endl;
        exit(1);
    }

    glfwMakeContextCurrent(window);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1);
    glMatrixMode(GL_MODELVIEW);

    return window;
}

GLuint* initTexture() {
    GLuint* texture = new GLuint();

    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvf(GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    return texture;
}

Pixel* initImage() {
    Pixel* image = (Pixel*)malloc(sizeof(Pixel) * WINDOW_HEIGHT * WINDOW_WIDTH);
    clearImage(image);
    return image;
}

void drawImage(Pixel* image, GLuint* texture) {
    glEnable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, *texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 image);
    glBegin(GL_QUADS);
    {
        int x = 0;
        int y = 0;

        glTexCoord2d(0, 0);
        glVertex2i(x, y);
        glTexCoord2d(1, 0);
        glVertex2i(x + WINDOW_WIDTH, y);
        glTexCoord2d(1, 1);
        glVertex2i(x + WINDOW_WIDTH, y + WINDOW_HEIGHT);
        glTexCoord2d(0, 1);
        glVertex2i(x, y + WINDOW_HEIGHT);
    }
    glEnd();
    glDisable(GL_TEXTURE_2D);
}
