cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
### no cygwin assumption
set(CMAKE_LEGACY_CYGWIN_WIN32 0)

project("Neural Networks on CPU")

enable_testing()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # to turn on more strict rules: set(warnings "-Wall -Wextra -Werror")
    set(warnings "-Wall")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(warnings "/W4 /WX /EHsc")
endif()
if (NOT CONFIGURED_ONCE)
    set(CMAKE_CXX_FLAGS "${warnings}"
        CACHE STRING "Flags used by the compiler during all build types." FORCE)
    set(CMAKE_C_FLAGS   "${warnings}"
        CACHE STRING "Flags used by the compiler during all build types." FORCE)
endif()

set(CONFIGURED_ONCE TRUE CACHE INTERNAL "Show that CMake has been set up.")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#add_subdirectory(unit_test)

### pthread
find_package(Threads)

### compiler flags
set(CMAKE_CXX_FLAGS "-Wall -O2 -D MT=1")

add_executable(nn nn.cpp)

target_link_libraries(nn
                      -lpthread)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
