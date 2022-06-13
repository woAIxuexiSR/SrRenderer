include_directories(Opengl)
include_directories(Intergrator)
include_directories(Model)

add_subdirectory(Opengl)
add_subdirectory(Model)
add_subdirectory(Intergrator)

list(APPEND PROJECT_LIBS window)
list(APPEND PROJECT_LIBS model)
list(APPEND PROJECT_LIBS intergrator)

add_executable(main main.cu ${PROJECT_SRCS})
target_link_libraries(main PUBLIC ${PROJECT_LIBS})