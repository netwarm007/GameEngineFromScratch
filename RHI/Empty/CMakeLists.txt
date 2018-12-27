add_library(EmptyRHI
EmptyGraphicsManager.cpp
EmptyShaderManager.cpp
)
target_include_directories(EmptyRHI PUBLIC GeomMath)
target_link_libraries(EmptyRHI GeomMath)
