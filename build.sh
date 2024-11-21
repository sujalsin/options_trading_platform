#!/bin/bash

# Create build directory
mkdir -p cpp/build
cd cpp/build

# Configure CMake
cmake ..

# Build
cmake --build . --config Release

# Copy the built module to the Python package directory
cp optionslib_cpp*.so ../../python/optionslib/

echo "Build complete! The C++ module has been built and copied to the Python package directory."
