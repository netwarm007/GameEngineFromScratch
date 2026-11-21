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
