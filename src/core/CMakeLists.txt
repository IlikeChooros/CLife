add_library(core "")

target_sources(core
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/NeuralNetwork.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Layer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Neuron.cpp
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/NeuralNetwork.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Layer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Neuron.hpp
  )

target_include_directories(core
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
  )
