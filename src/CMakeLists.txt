include_directories(GUI)
include_directories(Model)
include_directories(Intergrator)

add_subdirectory(GUI)
add_subdirectory(Model)
add_subdirectory(Intergrator)

list(APPEND PROJECT_LIBS gui)
list(APPEND PROJECT_LIBS model)
list(APPEND PROJECT_LIBS intergrator)


add_executable(main main.cu ${PROJECT_SRCS})
target_link_libraries(main PUBLIC ${PROJECT_LIBS})