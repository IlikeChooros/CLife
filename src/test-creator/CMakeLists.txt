
add_library(test-creator)

target_sources(
    test-creator 
    PRIVATE
        src/TestCreator.cpp   
)

target_include_directories(
    test-creator PUBLIC include
)

