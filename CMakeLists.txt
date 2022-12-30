cmake_minimum_required(VERSION 3.20)
project(Simulator)
enable_language(CUDA)
set (CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_package(OpenGL REQUIRED)
find_package(OpenMP)

link_directories(${OpenGL_LIBRARY})

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory("lib/glfw")
include_directories("./src")
include_directories("./src/atoms")
include_directories("./src/bonds")
include_directories("./src/draw")
include_directories("./src/gui")
include_directories("./src/simulator")
include_directories("./lib/glfw/include/GLFW")

add_compile_options(-g -O3 -Wall -Wextra -fopenmp)

add_executable(Simulator ./src/main.cpp 
                         ./src/atoms/atoms.cu 
                         ./src/atoms/atoms.cpp 
                         ./src/bonds/bonds.cu 
                         ./src/draw/draw.cu 
                         ./src/draw/draw.cpp
                         ./src/gui/gui.cpp) 

target_include_directories(
    Simulator 
    PRIVATE 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(Simulator ${CUDART_LIBRARY} glfw OpenGL::GL OpenMP::OpenMP_CXX)