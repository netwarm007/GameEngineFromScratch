add_library(D3dRHI
    D3d12GraphicsManager.cpp
    D3d12PipelineStateManager.cpp
)

target_link_libraries(D3dRHI
    d3d12
    dxgi
)
