add_library(core "")

target_include_directories(core
PUBLIC include
)

target_sources(core 
PRIVATE
    src/NeuralNetwork.cpp
    src/Layer.cpp
    src/Neuron.cpp
)


