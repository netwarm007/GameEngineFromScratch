add_library(PlatformInterface 
                EmptyApplication.cpp
)
add_library(MyGameEngine EmptyApplicationConfig.cpp)
target_link_libraries(MyGameEngine Common PlatformInterface)
