add_subdirectory(Opengl)
add_subdirectory(Model)
add_subdirectory(Intergrator)

list(APPEND PROJECT_LIBS window)

add_executable(main main.cu ${PROJECT_SRCS})
target_link_libraries(main PUBLIC ${PROJECT_LIBS})