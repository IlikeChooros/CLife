
add_library(test-creator)

target_sources(
    test-creator 
    PRIVATE
        src/TestCreator.cpp   
)

target_include_directories(
    test-creator PUBLIC include
)

target_link_libraries(test-creator data)
