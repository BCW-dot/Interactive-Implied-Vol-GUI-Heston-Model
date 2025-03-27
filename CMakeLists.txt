cmake_minimum_required(VERSION 3.10)
project(VolatilitySurfaceExplorer)

set(CMAKE_CXX_STANDARD 11)

# Source files
set(SOURCES
    src/main.cpp
    src/glad.c
    imgui/imgui.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_glfw.cpp
    imgui/backends/imgui_impl_opengl3.cpp
)

# Include directories
include_directories(
    include
    include/glad
    imgui
    imgui/backends
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} glfw GL dl)