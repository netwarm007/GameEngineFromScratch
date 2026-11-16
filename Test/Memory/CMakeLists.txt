cmake_minimum_required(VERSION 4.0)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(MemoryAllocatorRegression LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    include(CTest)
endif()

# Allocators are self-contained and are tested without the rest of the engine.
set(ALLOCATOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/Manager")
set(ALLOCATOR_SOURCES
    "${ALLOCATOR_DIR}/MemoryManager.cpp"
    "${ALLOCATOR_DIR}/BlockAllocator.cpp"
    "${ALLOCATOR_DIR}/LinearAllocator.cpp"
    "${ALLOCATOR_DIR}/FrameAllocator.cpp"
    "${ALLOCATOR_DIR}/RingAllocator.cpp"
    "${ALLOCATOR_DIR}/SizeClassAllocator.cpp"
    "${ALLOCATOR_DIR}/SlabAllocator.cpp"
    "${ALLOCATOR_DIR}/StackAllocator.cpp")

foreach(TEST_CASE MemoryAllocatorTest LinearAllocatorTest StackAllocatorTest
    FrameAllocatorTest RingAllocatorTest PmrAdapterTest SlabAllocatorTest
    SizeClassAllocatorTest)
    add_executable(${TEST_CASE} ${TEST_CASE}.cpp ${ALLOCATOR_SOURCES})
    target_include_directories(${TEST_CASE} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/Common"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/Interface"
        "${ALLOCATOR_DIR}")
    add_test(NAME TEST_${TEST_CASE} COMMAND ${TEST_CASE})
endforeach()
