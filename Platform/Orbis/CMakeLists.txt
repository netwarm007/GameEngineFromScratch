add_library(PlatformInterface
                OrbisApplication.cpp 
)

add_library(MyGameEngine
                OrbisApplicationConfig.cpp
    )
target_link_libraries(MyGameEngine
                        Common 
                        OrbisGnm
                        PlatformInterface
)

