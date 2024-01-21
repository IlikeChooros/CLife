
add_executable(${PROJECT_NAME} main.cpp)

add_subdirectory(data)
add_subdirectory(ui)
add_subdirectory(core)
add_subdirectory(backend)
add_subdirectory(test-creator)

target_link_libraries(
    ${PROJECT_NAME} 
        PUBLIC 
            data
            backend
            ui
            core
            test-creator
)

target_link_libraries(${PROJECT_NAME} PUBLIC sfml-graphics sfml-window sfml-system) 