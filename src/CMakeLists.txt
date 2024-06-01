
add_subdirectory(data)
add_subdirectory(mnist)
add_subdirectory(core)
add_subdirectory(backend)
add_subdirectory(test-creator)

add_subdirectory(optimizer)
add_subdirectory(games)
add_subdirectory(ui)


target_link_libraries(
    ${PROJECT_NAME} 
        PRIVATE 
            sfml-graphics sfml-window sfml-system sfml-audio
        PUBLIC
            mnist
            data
            backend
            ui
            core
            optimizer
            test-creator
            games
)

configure_file(${CMAKE_CURRENT_LIST_DIR}/Ubuntu-L.ttf ${CMAKE_BINARY_DIR} COPYONLY)
