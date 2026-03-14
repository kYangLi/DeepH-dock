cmake_minimum_required(VERSION 3.15)
project(overlap_openmx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 REQUIRED)

set(SOURCES
    src/bessel.cpp
    src/gaunt.cpp
    src/basis.cpp
    binding/pybind.cpp
)

pybind11_add_module(overlap_openmx ${SOURCES})

target_link_libraries(overlap_openmx PRIVATE
    Eigen3::Eigen
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
)

target_include_directories(overlap_openmx PRIVATE
    ${HDF5_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    include
)

target_compile_definitions(overlap_openmx PRIVATE
    ${HDF5_DEFINITIONS}
)

install(TARGETS overlap_openmx LIBRARY DESTINATION ".")
