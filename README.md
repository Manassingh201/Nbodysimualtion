# N-Body Simulation

A 3D N-body simulation using OpenGL, demonstrating gravitational interactions between multiple bodies in space.

## Prerequisites

Before running this program, you need to install the following dependencies:

### Windows (using vcpkg)
```bash
# Install vcpkg if you haven't already
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat

# Install required packages
.\vcpkg install glew:x64-windows
.\vcpkg install glfw3:x64-windows
.\vcpkg install glm:x64-windows
```