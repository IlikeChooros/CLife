add_library(data "")

target_sources(
    data PRIVATE
        src/Data.cpp   
        src/TestData.cpp   
        src/DoodlesLoader.cpp
)

target_include_directories(
    data PUBLIC include
)