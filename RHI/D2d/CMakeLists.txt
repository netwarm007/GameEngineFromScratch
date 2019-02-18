add_library(D2dRHI
    D2dGraphicsManager.cpp
)

target_link_libraries(D2dRHI
    d2d1
    dxgi
)
