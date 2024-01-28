add_library(mnist "")

add_subdirectory(digits)

target_sources(mnist
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src/loader.cpp
    )

target_include_directories(mnist
    PUBLIC include)

target_link_libraries(mnist 
    PUBLIC data) 