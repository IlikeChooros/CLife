add_library(games)

target_sources(games
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src/TicTacToe.cpp
    )

target_include_directories(games
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(
    games
    PUBLIC core mnist
)