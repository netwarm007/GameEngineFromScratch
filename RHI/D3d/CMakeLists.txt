add_library(D3dRHI
D3d12GraphicsManager.cpp
D3dShaderManager.cpp
)
target_include_directories(D3dRHI PUBLIC GeomMath)
target_link_libraries(D3dRHI GeomMath)
