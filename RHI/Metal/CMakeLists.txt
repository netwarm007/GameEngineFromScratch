find_library(MTK_LIBRARY MetalKit required)
find_library(METAL_LIBRARY Metal required)
find_library(QUARTZ_LIBRARY QuartzCore required)

MESSAGE( STATUS "MTK_LIBRARY: " ${MTK_LIBRARY} )

add_library(MetalRHI
    Metal2RHI.mm
    Metal2GraphicsManager.mm
    MetalPipelineStateManager.mm
    imgui_impl_metal.mm
    )
target_link_libraries(MetalRHI
        ${MTK_LIBRARY} 
        ${METAL_LIBRARY}
        ${QUARTZ_LIBRARY}
    )

set_property (TARGET MetalRHI APPEND_STRING PROPERTY 
COMPILE_FLAGS "-fobjc-weak")