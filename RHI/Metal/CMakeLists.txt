find_library(MTK_LIBRARY MetalKit required)
find_library(METAL_LIBRARY Metal required)

MESSAGE( STATUS "MTK_LIBRARY: " ${MTK_LIBRARY} )

add_library(MetalRHI
    Metal2Renderer.mm
    Metal2GraphicsManager.mm
    MetalShaderManager.mm
    )
target_include_directories(MetalRHI PUBLIC GeomMath)
target_link_libraries(MetalRHI
        GeomMath
        ${MTK_LIBRARY} 
        ${METAL_LIBRARY}
    )