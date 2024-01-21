add_executable(tests unit-tests.cpp)
add_library(testlib tests.cpp TestCase.cpp)


target_link_libraries(testlib PRIVATE core backend test-creator data)

target_link_libraries(tests PUBLIC testlib core backend test-creator data)

add_test(
  NAME
    all_tests
  COMMAND
    $<TARGET_FILE:tests>
  )