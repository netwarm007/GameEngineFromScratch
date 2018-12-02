add_library(DrawPhase 
ForwardRenderPhase.cpp
HUDPhase.cpp
SkyBoxPhase.cpp
TerrainPhase.cpp
)
target_include_directories(DrawPhase PUBLIC GeomMath)
target_link_libraries(DrawPhase PUBLIC GeomMath)