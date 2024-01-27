add_library(mnist "")

target_sources(mnist
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src/loader.cpp
    )

target_include_directories(mnist
    PUBLIC include)
