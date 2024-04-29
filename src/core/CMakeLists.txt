add_library(core "")

target_include_directories(core
PUBLIC include
)

target_sources(core 
PRIVATE
    src/OLayer.cpp
    src/ONeural.cpp
    src/activation.cpp
    src/LinearModel.cpp
    src/utils.cpp
    src/thread.cpp
    src/CNN.cpp
    src/ConvolutionLayer.cpp
    src/MaxPool.cpp
)

target_link_libraries(core PUBLIC data)
