add_library(optimizer)

target_include_directories(optimizer
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include
  )

target_sources(optimizer
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src/NeuralNetworkOptimizer.cpp
    )


target_link_libraries(
    optimizer
    PUBLIC core mnist ui
)