
add_executable(${PROJECT_NAME} main.cpp)

add_subdirectory(data)
add_subdirectory(core)
add_subdirectory(backend)
add_subdirectory(test-creator)

add_subdirectory(ui)

target_link_libraries(
    ${PROJECT_NAME} 
        PUBLIC 
            data
            backend
            ui
            core
            test-creator
)


configure_file(${CMAKE_SOURCE_DIR}/Ubuntu-L.ttf ${CMAKE_BINARY_DIR} COPYONLY)
target_link_libraries(${PROJECT_NAME} PUBLIC sfml-graphics sfml-window sfml-system) 