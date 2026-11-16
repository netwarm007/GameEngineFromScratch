cmake_minimum_required(VERSION 4.0)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(MemoryAllocatorRegression LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    include(CTest)
endif()

# Allocators are self-contained and are tested without the rest of the engine.
add_executable(MemoryAllocatorTest MemoryAllocatorTest.cpp
    "${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/Manager/MemoryManager.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/Manager/BlockAllocator.cpp")
target_include_directories(MemoryAllocatorTest PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/Common"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/Interface"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/Manager")
add_test(NAME TEST_MemoryAllocatorTest COMMAND MemoryAllocatorTest)
