cmake_minimum_required(VERSION 3.4)

# https://stackoverflow.com/questions/13703647/how-to-properly-add-include-directories-with-cmake

project(CLife)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall")

set(INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
set(TEST_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/test)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 2.6.x)
FetchContent_MakeAvailable(SFML)

add_subdirectory(src) 
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)


if(WIN32)
    add_custom_command(
        TARGET CMakeSFMLProject
        COMMENT "Copy OpenAL DLL"
        PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${SFML_SOURCE_DIR}/extlibs/bin/$<IF:$<BOOL:${ARCH_64BITS}>,x64,x86>/openal32.dll $<TARGET_FILE_DIR:CMakeSFMLProject>
        VERBATIM)
endif()

install(TARGETS ${PROJECT_NAME})

enable_testing()
add_subdirectory(tests)