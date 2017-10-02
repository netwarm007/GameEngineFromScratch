add_executable(AssetLoaderTest AssetLoaderTest.cpp)

target_link_libraries(AssetLoaderTest Common)

add_test(NAME TEST_AssetLoader COMMAND AssetLoaderTest)

