add_library(PlatformInterface 
                EmptyApplication.cpp
)
add_executable(Empty EmptyApplicationConfig.cpp)
target_link_libraries(Empty Common PlatformInterface)
__add_xg_platform_dependencies(Empty)
