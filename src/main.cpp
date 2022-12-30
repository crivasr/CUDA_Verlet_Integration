#include <iostream>
#include <thread>
#include <vector>

#include "../lib/glfw/include/GLFW/glfw3.h"
#include "./atoms/atoms.h"
#include "./bonds/bonds.h"
#include "./draw/draw.h"
#include "./gui/gui.h"
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

    std::vector<Atom> atoms;
    std::vector<Bond> bonds;

    int width = 100;
    int height = 100;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float x = (float)j * (600 / width) + 50;
            float y = (float)650 - i * (400 / height);

            Atom atom = {x, y, x, y, false};
            atoms.push_back(atom);
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

            Bond bond = {idxA, idxB, dst, false};
            bonds.push_back(bond);
        }
    }

    for (int i = 0; i < height - 1; i++) {
        for (int j = 0; j < width; j++) {
            int idxA = j + i * width;
            int idxB = j + (i + 1) * width;

            Atom* a = &atoms[idxA];
            Atom* b = &atoms[idxB];

            float dst = distance(a, b);

            Bond bond = {idxA, idxB, dst, false};
            bonds.push_back(bond);
        }
    }

    std::thread events(eventThread, window);

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        Atom* atomsArray = atoms.data();
        Bond* bondsArray = bonds.data();

        int atomsSize = atoms.size();
        int bondsSize = bonds.size();

        updateAtoms(atomsArray, atomsSize);

        for (int _ = 0; _ < BOND_UPDATES; _++) {
            updateBonds(bondsArray, atomsArray, bondsSize, atomsSize);
        }

        constrainAtoms(atomsArray, atomsSize);

        drawAtoms(image, atomsArray, atomsSize);
        drawBonds(image, bondsArray, atomsArray, bondsSize, atomsSize);

        drawImage(image, texture);
        clearImage(image);

        glfwSwapBuffers(window);
    };
    events.join();
}