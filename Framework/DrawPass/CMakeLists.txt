add_library(DrawPass 
ForwardRenderPass.cpp
ShadowMapPass.cpp
HUDPass.cpp
)
target_include_directories(DrawPass PUBLIC GeomMath)
target_link_libraries(DrawPass PUBLIC GeomMath)