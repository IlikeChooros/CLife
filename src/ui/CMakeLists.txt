add_library(ui "")

target_sources(ui
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/windowWithDrawer.cpp
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/windowWithDrawer.hpp
  )

target_include_directories(ui
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
  )

target_link_libraries(ui sfml-graphics sfml-window sfml-system core)