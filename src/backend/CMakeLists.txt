add_library(backend "")

target_include_directories(backend
    PUBLIC include
)

target_sources(backend 
PRIVATE
    src/Database.cpp
    src/FileManager.cpp
)

target_link_libraries(backend PUBLIC core nlohmann_json::nlohmann_json)