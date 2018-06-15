add_library(PlatformInterface 
                EmptyApplication.cpp
)
target_include_directories(PlatformInterface PUBLIC GeomMath)
target_link_libraries(PlatformInterface GeomMath)
add_library(MyGameEngine EmptyApplicationConfig.cpp)
target_link_libraries(MyGameEngine 
            Common 
            PlatformInterface
)
