cuda_compile_and_embed(simple "shader/test.cu")

add_library(intergrator STATIC Intergrator.cpp ${embedded_file})