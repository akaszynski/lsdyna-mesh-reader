cmake_minimum_required(VERSION 3.15...3.26)

project(nanobind_project LANGUAGES CXX)
# find_package(OpenMP REQUIRED)

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_deck STABLE_ABI NB_STATIC src/deck.cpp)

# Compiler-specific options
if(MSVC)
  # Use MSVC optimization levels and OpenMP setup
  target_compile_options(_deck PRIVATE /O2 /std:c++17)
else()
  # Assuming GCC or Clang
  target_compile_options(_deck PRIVATE -O3)
endif()

# Example debugging
# set solib-search-path /home/user/python/.venv311/lib/python3.11/site-packages/lsdyna_mesh_reader/
# set breakpoint with b _deck.cpp:<LINE_NUMBER>
# target_compile_options(_deck PRIVATE -g -O0)

# Install directive for scikit-build-core
install(TARGETS _deck LIBRARY DESTINATION lsdyna_mesh_reader)
