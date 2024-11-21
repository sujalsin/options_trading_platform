cmake_minimum_required(VERSION 3.15)
project(optionslib_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Set pybind11 paths
set(PYBIND11_INCLUDE_DIR "/usr/local/include")
set(PYBIND11_PYTHON_INCLUDE_DIR "/opt/anaconda3/include/python3.8")

# Add the optionslib library
add_library(optionslib STATIC
    optionslib/options_pricing.cpp
)

target_include_directories(optionslib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYBIND11_INCLUDE_DIR}
    ${PYBIND11_PYTHON_INCLUDE_DIR}
)

# Add the Python module
add_library(optionslib_cpp MODULE
    optionslib/bindings.cpp
)

target_include_directories(optionslib_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYBIND11_INCLUDE_DIR}
    ${PYBIND11_PYTHON_INCLUDE_DIR}
)

target_link_libraries(optionslib_cpp PRIVATE 
    optionslib
    Python::Python
)

# Set output name
set_target_properties(optionslib_cpp PROPERTIES 
    PREFIX ""
    SUFFIX ".so"
)
