cmake_minimum_required(VERSION 3.5)

project(sphere)

set(CMAKE_CXX_FLAGS "-Wall -Werror -std=c++14")

find_package(Eigen3 REQUIRED)

configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src ${CMAKE_BINARY_DIR}/googletest-build)

include_directories(${EIGEN3_INCLUDE_DIR})

SET(SOURCE_FILES src/PythongraphicsFramework.cpp)

add_executable(sphere src/main.cpp ${SOURCE_FILES})
add_executable(run_tests src/tests.cpp ${SOURCE_FILES})

target_link_libraries(run_tests gtest_main)
