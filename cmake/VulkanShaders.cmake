set(MYGE_SHARED_SHADER_SOURCES
    simple.vert simple.frag
    basic.vert basic.frag
    debug.vert debug.frag
    cubemap.frag cubemaparray.frag
    pbr.vert pbr.frag
    skybox.vert skybox.frag
    shadowmap.vert shadowmap.frag
    shadowmap_omni.vert shadowmap_omni.frag
    texture.frag texturearray.frag
    passthrough.vert passthrough_cube.vert
    integrateBRDF.comp raytrace.comp)

set(MYGE_VULKAN_GLSLANG_OPTIONS -V --target-env vulkan1.1
    --define-macro MYGE_GLSLANG=1
    --define-macro MYGE_VULKAN=1
    --shift-UBO-binding 0 --shift-texture-binding 32
    --shift-sampler-binding 64 --shift-image-binding 96 --shift-uav-binding 96)

function(myge_add_vulkan_shaders target root output config_dir)
    set(shaders ${MYGE_SHARED_SHADER_SOURCES}
        shadowmap_omni.geom shadowmap_omni_metal.vert)
    file(GLOB headers CONFIGURE_DEPENDS "${root}/Asset/Shaders/HLSL/*.hlsl")
    set(outputs)
    foreach(shader IN LISTS shaders)
        string(REPLACE "." "_" entry "${shader}")
        set(binary "${output}/${shader}.spv")
        add_custom_command(OUTPUT "${binary}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${output}"
            COMMAND "${GLSL_VALIDATOR}" ${MYGE_VULKAN_GLSLANG_OPTIONS}
                "-I${config_dir}" "-I${root}/Framework/Common"
                -e "${entry}_main" -o "${binary}"
                "${root}/Asset/Shaders/HLSL/${shader}.hlsl"
            DEPENDS ${headers} "${config_dir}/config.h"
                "${root}/Framework/Common/cbuffer.h"
                "${root}/Framework/Common/root_signature.h.hlsl"
            VERBATIM)
        list(APPEND outputs "${binary}")
    endforeach()
    set(single_face "${output}/shadowmap_omni_single.vert.spv")
    add_custom_command(OUTPUT "${single_face}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${output}"
        COMMAND "${GLSL_VALIDATOR}" ${MYGE_VULKAN_GLSLANG_OPTIONS}
            --define-macro MYGE_POINT_SHADOW_SINGLE_FACE=1
            "-I${config_dir}" "-I${root}/Framework/Common"
            -e shadowmap_omni_single_vert_main -o "${single_face}"
            "${root}/Asset/Shaders/HLSL/shadowmap_omni.vert.hlsl"
        DEPENDS ${headers} "${config_dir}/config.h"
            "${root}/Framework/Common/cbuffer.h"
            "${root}/Framework/Common/root_signature.h.hlsl"
        VERBATIM)
    list(APPEND outputs "${single_face}")
    list(APPEND shaders shadowmap_omni_single.vert)
    add_custom_target(${target} DEPENDS ${outputs})
    set(MYGE_VULKAN_SHADER_SOURCES "${shaders}" PARENT_SCOPE)
endfunction()
