cmake_minimum_required(VERSION 3)

project(chemistry)

set(CMAKE_CXX_FLAGS "-std=c++14 -Werror -fopenmp -DSPDLOG_DEBUG_ON")
set(CMAKE_EXE_LINKER_FLAGS "-pthread")
SET(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

find_package(Eigen3 REQUIRED)
#find_package(Boost REQUIRED COMPONENTS system filesystem)

# gtest
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


#spdlog
if (NOT EXISTS "${CMAKE_SOURCE_DIR}/thirdparty/spdlog/include")
    message(SEND_ERROR "CImg library submodule was not downloaded! Please, execute: git submodule update --init")
endif()
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/spdlog/include)

include_directories(${SOURCE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

#include_directories(${SRC)
SET(SOURCE_FILES
        src/helper.cpp
        src/InputOutputUtils.cpp
        src/linearAlgebraUtils.cpp
        src/testing/coordinateSystemChange.cpp
        src/producers/FixValues.cpp
        src/producers/FunctionProducer.cpp)

add_executable(chemistry ${SOURCE_DIR}/main.cpp ${SOURCE_FILES})
target_link_libraries(chemistry ${Boost_LIBRARIES})

add_executable(run_tests ${SOURCE_DIR}/tests.cpp ${SOURCE_FILES})
target_link_libraries(run_tests gtest_main )
