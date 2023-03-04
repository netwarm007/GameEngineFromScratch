add_library(D2dRHI
    D2dRHI.cpp
)

target_link_libraries(D2dRHI
    d2d1
    dxgi
)
