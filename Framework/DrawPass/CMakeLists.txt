add_library(DrawPass 
ForwardRenderPass.cpp
HUDPass.cpp
ShadowMapPass.cpp
SkyBoxPass.cpp
)
target_include_directories(DrawPass PUBLIC GeomMath)
target_link_libraries(DrawPass PUBLIC GeomMath)