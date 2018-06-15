add_library(D2dRHI
D2dGraphicsManager.cpp
)
target_include_directories(D2dRHI PUBLIC GeomMath)
target_link_libraries(D2dRHI GeomMath)
