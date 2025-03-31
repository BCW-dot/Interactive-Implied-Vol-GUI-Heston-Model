cmake_minimum_required(VERSION 3.16)
project(VolatilitySurfaceExplorer LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable AVX, AMX and other advanced instruction sets that might be causing issues
add_compile_options(-fno-tree-vectorize)

# Set the path to your local GLFW installation
set(GLFW_DIR $ENV{HOME}/local/glfw)

# Source files for the visualization part
set(SOURCES
    src/main.cpp
    src/glad.c
    src/grid.cpp
    src/base_prices.cpp
    imgui/imgui.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_glfw.cpp
    imgui/backends/imgui_impl_opengl3.cpp
)

# Include directories for visualization
include_directories(
    include
    include/glad
    imgui
    imgui/backends
    ${GLFW_DIR}/include 
)

add_definitions(-DIMGUI_DISABLE_SSE -DIMGUI_DISABLE_SSE2 -DIMGUI_DISABLE_AMXTILEINTRIN)

# Set Kokkos paths
set(Kokkos_DIR /home/bcwilkens/software/kokkos-install/lib64/cmake/Kokkos)
set(KokkosKernels_DIR /home/bcwilkens/software/kokkos-kernels-install/lib64/cmake/KokkosKernels)
list(APPEND CMAKE_PREFIX_PATH ${Kokkos_ROOT} ${KokkosKernels_ROOT})

# Configure Kokkos options
set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "")
set(Kokkos_ENABLE_CUDA ON CACHE BOOL "")
set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "")
set(Kokkos_ARCH_AMPERE80 ON CACHE BOOL "")  # For A100 GPU


# Find Kokkos packages
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include Kokkos directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${KokkosKernels_ROOT}/include
)

# Link libraries - visualization first, then Kokkos
target_link_libraries(${PROJECT_NAME} 
    ${GLFW_DIR}/lib64/libglfw3.a
    GL dl X11 Xrandr Xinerama Xcursor Xi Xxf86vm pthread 
    Kokkos::kokkos
    KokkosKernels::kokkoskernels
)

# If CUDA is enabled, set CUDA-specific properties
if(Kokkos_ENABLE_CUDA)
    message(STATUS "CUDA enabled for Kokkos")
    enable_language(CUDA)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 80)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# Set compiler flags to disable problematic intrinsics
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-march=x86-64 -mno-avx -mno-avx2 -mno-amx)
    add_compile_options(-DIMGUI_DISABLE_SSE -DIMGUI_DISABLE_SSE2)
    # Add this line to disable the AMX intrinsics
    add_compile_options(-DNO_X86INTRIN -DIMGUI_DISABLE_AMXTILEINTRIN)
    message(STATUS "Disabled AVX, AVX2, and AMX instructions")
endif()

# Print debug info
message(STATUS "Executable will be built in: ${CMAKE_BINARY_DIR}")
message(STATUS "Kokkos include path: ${Kokkos_ROOT}/include")
message(STATUS "KokkosKernels include path: ${KokkosKernels_ROOT}/include")
