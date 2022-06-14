add_library(intergrator STATIC Intergrator.cpp)

add_library(test OBJECT test.cu)
target_link_libraries(test PUBLIC model)
target_include_directories(test PUBLIC .)
set_property(TARGET test PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET test PROPERTY CUDA_ARCHITECTURES 86-virtual)
# file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/ptx")
# file(COPY_FILE "${CMAKE_BINARY_DIR}/src/Intergrator/test.dir/${CMAKE_BUILD_TYPE}/test.ptx" "${CMAKE_BINARY_DIR}/ptx/test.ptx")

# find_package(CUDA REQUIRED)


# find_program(BIN2C bin2c
#   DOC "Path to the cuda-sdk bin2c executable.")

# set(output_var embedded_ptx_code)
# cuda_compile_ptx(ptx_files main.cu OPTIONS --generate-line-info -use_fast_math --keep)
# list(GET ptx_files 0 ptx_file)
# message("ptx_fiels : " ${ptx_files})
# message(${ptx_file})
# set(embedded_file ${ptx_file}_embedded.c)
# add_custom_command(
#     OUTPUT ${embedded_file}
#     COMMAND ${BIN2C} -c --padd 0 --type char --name ${output_var} ${ptx_file} > ${embedded_file}
#     DEPENDS ${ptx_file}
# )
# message(${embedded_file})