add_library(ui "")

target_sources(ui
  PRIVATE
    src/windowWithDrawer.cpp
    src/Drawer.cpp
    src/Visualizer.cpp
  )

target_include_directories(ui
  PUBLIC include
  )

target_link_libraries(ui sfml-graphics sfml-window sfml-system core test-creator backend mnist)