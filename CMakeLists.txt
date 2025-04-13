cmake_minimum_required(VERSION 3.10)
project(NBodySimulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

# Add executable
add_executable(nbody Nbody.cpp)

# Link libraries
target_link_libraries(nbody PRIVATE
    OpenGL::GL
    GLEW::GLEW
    glfw
    glm::glm
)

# Include directories
target_include_directories(nbody PRIVATE
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIRS}
) 