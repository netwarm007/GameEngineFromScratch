add_platform_asset_io(PlatformAssetIO AssetLoader.cpp)

add_library(PlatformInterface 
                EmptyApplication.cpp
)
target_link_libraries(PlatformInterface PUBLIC PlatformAssetIO)
