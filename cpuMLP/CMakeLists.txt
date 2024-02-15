cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(PROJECT_NAMESPACE gepetto)
set(PROJECT_NAME cpp-mlp)
set(PROJECT_DESCRIPTION "Reinforcement learning for Solo")
set(PROJECT_URL https://gitlab.laas.fr/${PROJECT_NAMESPACE}/${PROJECT_NAME})

project(${PROJECT_NAME} CXX)

SET(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_PREFIX_PATH libtorch ${CMAKE_PREFIX_PATH})

add_subdirectory(pybind11 pybind11)
include_directories("eigen3" ${TORCH_INCLUDE_DIRS})

pybind11_add_module(policy cpuMLP.cpp cpuMLP.hpp)
pybind11_add_module(stateEst cpuMLP.cpp cpuMLP.hpp)
pybind11_add_module(stateEst3 cpuMLP.cpp cpuMLP.hpp)
pybind11_add_module(interface cpuMLP.cpp cpuMLP.hpp)

add_executable(${PROJECT_NAME} main.cpp)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
set_property(TARGET policy PROPERTY CXX_STANDARD 14)
set_property(TARGET stateEst PROPERTY CXX_STANDARD 14)
set_property(TARGET stateEst3 PROPERTY CXX_STANDARD 14)
set_property(TARGET interface PROPERTY CXX_STANDARD 14)
