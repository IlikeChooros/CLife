
add_executable(${PROJECT_NAME} main.cpp)

# add_subdirectory(backend)
# add_subdirectory(tests)
add_subdirectory(ui)
add_subdirectory(core)

target_link_libraries(
    ${PROJECT_NAME} 
        PRIVATE 
            # tests
            # backend
            ui
            core
            
)

target_link_libraries(${PROJECT_NAME} PRIVATE sfml-graphics sfml-window sfml-system) 
