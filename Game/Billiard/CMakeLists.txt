IF(ANDROID)
add_library(BilliardGame SHARED BilliardGameLogic.cpp BilliardGameConfig.cpp)
ELSE(ANDROID)
add_executable(BilliardGame BilliardGameLogic.cpp BilliardGameConfig.cpp)
ENDIF(ANDROID)
target_link_libraries(BilliardGame 
                        MyGameEngine
                     )