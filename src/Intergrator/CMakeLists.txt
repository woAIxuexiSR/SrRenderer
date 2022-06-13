add_library(intergrator STATIC Intergrator.cpp)

add_library(test OBJECT test.cu)
target_link_libraries(test PUBLIC model)
target_include_directories(test PUBLIC .)
set_property(TARGET test PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET test PROPERTY CUDA_ARCHITECTURES 86-virtual)
# file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/ptx")
# file(COPY_FILE "${CMAKE_BINARY_DIR}/src/Intergrator/test.dir/${CMAKE_BUILD_TYPE}/test.ptx" "${CMAKE_BINARY_DIR}/ptx/test.ptx")