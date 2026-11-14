cmake_minimum_required(VERSION 4.0)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(RuntimeRegression LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    include(CTest)
endif()

add_executable(RuntimeModuleLifecycleTest RuntimeModuleLifecycleTest.cpp)
target_include_directories(RuntimeModuleLifecycleTest PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/Common"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/Interface"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/Manager")
add_test(NAME TEST_RuntimeModuleLifecycleTest COMMAND RuntimeModuleLifecycleTest)
