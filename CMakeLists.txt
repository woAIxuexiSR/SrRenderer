cmake_minimum_required(VERSION 3.10)

project(SrRenderer LANGUAGES C CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 86)

# glad and glfw
list(APPEND PROJECT_INCLUDES "${PROJECT_SOURCE_DIR}/ext/include")
list(APPEND PROJECT_LIBS "${PROJECT_SOURCE_DIR}/ext/lib/glfw3.lib")
list(APPEND PROJECT_SRCS "${PROJECT_SOURCE_DIR}/ext/src/glad.c")

# imgui
set(IMGUI_INSTALL_DIR "D:/software/imgui-master" CACHE PATH "Path to imgui")
file(GLOB IMGUI_SRC "${IMGUI_INSTALL_DIR}/*.cpp")
list(APPEND IMGUI_SRC "${IMGUI_INSTALL_DIR}/backends/imgui_impl_opengl3.cpp" "${IMGUI_INSTALL_DIR}/backends/imgui_impl_glfw.cpp")
list(APPEND PROJECT_INCLUDES "${IMGUI_INSTALL_DIR}" "${IMGUI_INSTALL_DIR}/backends")
list(APPEND PROJECT_SRCS "${IMGUI_SRC}")

# cuda
set(CUDA_SDK_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5" CACHE PATH "Path to CUDA SDK")
find_package(CUDA REQUIRED)
list(APPEND PROJECT_INCLUDES "${CUDA_TOOLKIT_ROOT_DIR}/include")
list(APPEND PROJECT_LIBS "${CUDA_LIBRARIES}" "${CUDA_CUDA_LIBRARY}" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nvrtc.lib")

# optix
set(OptiX_INSTALL_DIR "D:/software/OptiX SDK 7.4.0" CACHE PATH "Path to OptiX SDK")
list(APPEND CMAKE_MODULE_PATH "${OptiX_INSTALL_DIR}/SDK/CMake")
find_package(OptiX REQUIRED)
list(APPEND PROJECT_INCLUDES ${OptiX_INCLUDE})


include_directories(${PROJECT_INCLUDES})

add_subdirectory(src)