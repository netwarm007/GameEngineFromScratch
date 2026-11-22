cmake_minimum_required(VERSION 4.0)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(RuntimeRegression LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    include(CTest)
    option(VULKAN_LOADER_TESTS "Test late Vulkan loading (headers only, no GPU)" OFF)
endif()

add_executable(RuntimeModuleLifecycleTest RuntimeModuleLifecycleTest.cpp)
target_include_directories(RuntimeModuleLifecycleTest PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/Common"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/Interface"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/Manager")
add_test(NAME TEST_RuntimeModuleLifecycleTest COMMAND RuntimeModuleLifecycleTest)

if(NOT TARGET RhiStartup)
    add_library(RhiStartup ../../Framework/Manager/RhiStartup.cpp)
    target_include_directories(RhiStartup PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/Manager")
endif()
add_executable(RhiStartupTest RhiStartupTest.cpp)
target_link_libraries(RhiStartupTest PRIVATE RhiStartup)
add_test(NAME TEST_RhiStartupTest COMMAND RhiStartupTest)
set_tests_properties(TEST_RhiStartupTest PROPERTIES LABELS fast)

if(TARGET VulkanLoader OR VULKAN_LOADER_TESTS)
    if(NOT TARGET VulkanLoader)
        find_path(Vulkan_INCLUDE_DIR vulkan/vulkan_core.h REQUIRED)
        get_filename_component(root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
        include("${root}/cmake/VulkanLoader.cmake")
        myge_add_vulkan_loader("${root}")
    endif()
    foreach(fixture VulkanLoaderFixture VulkanLoaderNoSymbol VulkanLoaderNoCreate)
        add_library(${fixture} SHARED VulkanLoaderFixture.cpp)
        target_include_directories(${fixture} PRIVATE "${Vulkan_INCLUDE_DIR}")
        target_compile_definitions(${fixture} PRIVATE VK_NO_PROTOTYPES)
    endforeach()
    target_compile_definitions(VulkanLoaderNoSymbol PRIVATE MYGE_FIXTURE_NO_SYMBOL)
    target_compile_definitions(VulkanLoaderNoCreate PRIVATE MYGE_FIXTURE_NO_CREATE)
    add_executable(VulkanLoaderTest VulkanLoaderTest.cpp)
    target_link_libraries(VulkanLoaderTest PRIVATE VulkanLoader RhiStartup)
    add_dependencies(VulkanLoaderTest
        VulkanLoaderFixture VulkanLoaderNoSymbol VulkanLoaderNoCreate)
    add_test(NAME TEST_VulkanLoaderTest COMMAND VulkanLoaderTest
        "$<TARGET_FILE:VulkanLoaderFixture>"
        "$<TARGET_FILE:VulkanLoaderNoSymbol>"
        "$<TARGET_FILE:VulkanLoaderNoCreate>")
    set_tests_properties(TEST_VulkanLoaderTest PROPERTIES LABELS fast TIMEOUT 10)
endif()

if(TARGET RuntimeComposition)
    add_executable(RhiCompositionTest RhiCompositionTest.cpp)
    target_link_libraries(RhiCompositionTest PRIVATE RuntimeComposition)
    add_test(NAME TEST_RhiCompositionTest COMMAND RhiCompositionTest)
    set_tests_properties(TEST_RhiCompositionTest PROPERTIES LABELS fast TIMEOUT 30)

    add_executable(RhiSwitchTest RhiSwitchTest.cpp)
    target_link_libraries(RhiSwitchTest PRIVATE RuntimeComposition)
    add_test(NAME TEST_RhiSwitchTest COMMAND RhiSwitchTest)
    set_tests_properties(TEST_RhiSwitchTest PROPERTIES LABELS fast TIMEOUT 30)
endif()

# M3 session contract: the persistent root lives in Manager for full builds;
# standalone regression builds compile its allocator closure directly.
if(TARGET Manager)
    add_executable(RenderSessionTest RenderSessionTest.cpp)
    target_link_libraries(RenderSessionTest PRIVATE Manager)
else()
    add_library(RuntimeRootDeps STATIC
        ../../Framework/Manager/RuntimeRoot.cpp
        ../../Framework/Manager/MemorySystem.cpp
        ../../Framework/Manager/EngineAllocation.cpp
        ../../Framework/Manager/MemoryManager.cpp
        ../../Framework/Manager/RoutingAllocator.cpp
        ../../Framework/Manager/ThreadCacheAllocator.cpp
        ../../Framework/Manager/TlsfAllocator.cpp
        ../../Framework/Manager/SizeClassAllocator.cpp
        ../../Framework/Manager/SlabAllocator.cpp
        ../../Framework/Manager/FrameAllocator.cpp
        ../../Framework/Manager/LinearAllocator.cpp
        ../../Framework/Manager/TrackingAllocator.cpp)
    target_include_directories(RuntimeRootDeps PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/Common"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/Interface"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../Framework/Manager")
    add_executable(RenderSessionTest RenderSessionTest.cpp)
    target_link_libraries(RenderSessionTest PRIVATE RuntimeRootDeps)
endif()
add_test(NAME TEST_RenderSessionTest COMMAND RenderSessionTest)
set_tests_properties(TEST_RenderSessionTest PROPERTIES LABELS fast TIMEOUT 30)
