#include <iostream>
#include <thread>
#include <vector>

#include "GLFW/glfw3.h"
#include "atoms/atoms.cuh"
#include "bonds/bonds.cuh"
#include "draw/draw.cuh"
#include "gui/gui.h"
#include "defines.h"

void eventThread(GLFWwindow* window) {
    while (!glfwWindowShouldClose(window)) {
        glfwWaitEvents();
    };
}

int main() {
    GLFWwindow* window = initWindow();
    GLuint* texture = initTexture();
    Pixel* image = initImage();

    int width = 300;
    int height = 300;

    int atomsSize = width*height;
    int bondsSize = (width-1)*height + width*(height-1);

    Atom* atoms;
    Bond* bonds;

    cudaMallocManaged((void**)&atoms, atomsSize * sizeof(Atom));
    cudaMallocManaged((void**)&bonds, bondsSize * sizeof(Bond));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float x = (float)j * (600 / width) + 50;
            float y = (float)650 - i * (400 / height);

            atoms[i * width + j] = atom(x, y);
        }
    }

    for (int i = 0; i < width / 3; i++) {
        (&atoms[i])->fixed = true;
        (&atoms[width - i - 1])->fixed = true;
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width - 1; j++) {
            int idxA = j + i * width;
            int idxB = j + i * width + 1;

            Atom* a = &atoms[idxA];
            Atom* b = &atoms[idxB];

            float dst = distance(a, b);

            bonds[i * (width - 1) + j] = bond(idxA, idxB, dst);
        }
    }

    for (int i = 0; i < height - 1; i++) {
        for (int j = 0; j < width; j++) {
            int idxA = j + i * width;
            int idxB = j + (i + 1) * width;

            Atom* a = &atoms[idxA];
            Atom* b = &atoms[idxB];

            float dst = distance(a, b);

            bonds[(height - 1) * (width - 1) + i * width + j] = bond(idxA, idxB, dst);
        }
    }

    std::thread events(eventThread, window);

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        for (int _ = 0; _ < SUB_STEPS; _++) {
            updateAtoms(atoms, atomsSize);

            preSolve(atoms, atomsSize);
            updateBonds(bonds, atoms, bondsSize);
            postSolve(atoms, atomsSize);

            constrainAtoms(atoms, atomsSize);
            updateVelocity(atoms, atomsSize);
        }
        
        // drawAtoms(image, atoms, atomsSize);
        drawBonds(image, bonds, atoms, bondsSize);

        drawImage(image, texture);
        clearImage(image);

        glfwSwapBuffers(window);
    };
    events.join();
}