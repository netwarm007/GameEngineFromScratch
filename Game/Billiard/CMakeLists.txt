IF(ANDROID)
add_library(BilliardGame SHARED 
    BilliardGameLogic.cpp BilliardGameConfig.cpp)
ELSE(ANDROID)
add_executable(BilliardGame MACOSX_BUNDLE
    BilliardGameLogic.cpp BilliardGameConfig.cpp)
ENDIF(ANDROID)
target_link_libraries(BilliardGame 
                        MyGameEngine
                     )