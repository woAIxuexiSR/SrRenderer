cmake_minimum_required(VERSION 3.10)

project(SrRenderer LANGUAGES C CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
# add_compile_definitions(-DNOMINMAX)
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

# implot
set(IMPLOT_INSTALL_DIR "D:/software/implot-master" CACHE PATH "Path to implot")
file(GLOB IMPLOT_SRC "${IMPLOT_INSTALL_DIR}/*.cpp")
list(APPEND PROJECT_INCLUDES "${IMPLOT_INSTALL_DIR}")
list(APPEND PROJECT_SRCS "${IMPLOT_SRC}")

# cuda
set(CUDA_SDK_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5" CACHE PATH "Path to CUDA SDK")
find_package(CUDA REQUIRED)
find_program(BIN2C bin2c)
list(APPEND PROJECT_INCLUDES "${CUDA_TOOLKIT_ROOT_DIR}/include")
list(APPEND PROJECT_LIBS "${CUDA_LIBRARIES}" "${CUDA_CUDA_LIBRARY}")

# cuda samples
set(CUDA_SAMPLES_DIR "D:/software/cuda-samples-master" CACHE PATH "Path to cuda samples")
list(APPEND PROJECT_INCLUDES "${CUDA_SAMPLES_DIR}/Common")

# optix
set(OptiX_INSTALL_DIR "D:/software/OptiX SDK 7.4.0" CACHE PATH "Path to OptiX SDK")
list(APPEND CMAKE_MODULE_PATH "${OptiX_INSTALL_DIR}/SDK/CMake")
find_package(OptiX REQUIRED)
list(APPEND PROJECT_INCLUDES ${OptiX_INCLUDE})

# macro for compile ptx file
macro(cuda_compile_and_embed output_var cuda_file)
  set(c_var_name ${output_var})
  cuda_compile_ptx(ptx_file ${cuda_file} OPTIONS --generate-line-info -use_fast_math --gpu-architecture=compute_86)
  set(embedded_file ${ptx_file}_embedded.c)
  add_custom_command(
    OUTPUT ${embedded_file}
    COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
    DEPENDS ${ptx_file}
    COMMENT "compiling (and embedding ptx from) ${cuda_file}"
    )
endmacro()


include_directories(${PROJECT_INCLUDES})

add_subdirectory(src)