add_custom_target(Engine_Asset_Shaders
        COMMAND echo "Start processing Engine Shaders"
    )

set(HLSL_SHADER_SOURCES 
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
            integrateBRDF.comp
            raytrace.comp
        )

set(GLSL_SHADER_SOURCES #simple.vert simple.frag
        )

IF(WIN32)
    set(GLSL_VALIDATOR ${PROJECT_SOURCE_DIR}/External/Windows/bin/glslangValidator.exe)
    set(SPIRV_CROSS    ${PROJECT_SOURCE_DIR}/External/Windows/bin/SPIRV-Cross.exe)
    set(DXC            dxc.exe)
ELSE(WIN32)
    if(CMAKE_CROSSCOMPILING)
        set(GLSL_VALIDATOR ${PROJECT_SOURCE_DIR}/External/${CMAKE_HOST_SYSTEM_NAME}/bin/glslangValidator)
        set(SPIRV_CROSS    ${PROJECT_SOURCE_DIR}/External/${CMAKE_HOST_SYSTEM_NAME}/bin/spirv-cross)
    else()
        set(GLSL_VALIDATOR ${PROJECT_SOURCE_DIR}/External/${MYGE_TARGET_PLATFORM}/bin/glslangValidator)
        set(SPIRV_CROSS    ${PROJECT_SOURCE_DIR}/External/${MYGE_TARGET_PLATFORM}/bin/spirv-cross)
    endif(CMAKE_CROSSCOMPILING)
ENDIF(WIN32)

set(VULKAN_SOURCE_DIR ${PROJECT_BINARY_DIR}/Asset/Shaders/Vulkan)
set(GLSL_SOURCE_DIR ${PROJECT_BINARY_DIR}/Asset/Shaders/OpenGL)
set(METAL_SOURCE_DIR ${PROJECT_BINARY_DIR}/Asset/Shaders/Metal)

add_custom_command(TARGET Engine_Asset_Shaders PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${VULKAN_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GLSL_SOURCE_DIR}
)

foreach(SHADER IN LISTS HLSL_SHADER_SOURCES)
    # Convert HLSL to Others
    string(REPLACE "." ";" arguments ${SHADER})
    list(GET arguments 0 part1)
    list(GET arguments 1 part2)

    add_custom_command(TARGET Engine_Asset_Shaders PRE_BUILD
        COMMENT "HLSL --> SPIR-V"
	    COMMAND ${GLSL_VALIDATOR} -V -I. -I${PROJECT_SOURCE_DIR}/Framework/Common -o ${VULKAN_SOURCE_DIR}/${part1}.${part2}.spv -e ${part1}_${part2}_main --uniform-base 1 ${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL/${part1}.${part2}.hlsl
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        DEPENDS HLSL/${part1}.${part2}.hlsl
    )
        
    add_custom_command(TARGET Engine_Asset_Shaders PRE_BUILD
        COMMENT "SPIR-V --> Desktop GLSL"
	    COMMAND ${SPIRV_CROSS} --version 420 --remove-unused-variables --output ${GLSL_SOURCE_DIR}/${part1}.${part2}.glsl ${PROJECT_BINARY_DIR}/Asset/Shaders/Vulkan/${part1}.${part2}.spv
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        DEPENDS HLSL/${part1}.${part2}.hlsl
    )

if(APPLE)
    add_custom_command(TARGET Engine_Asset_Shaders PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${METAL_SOURCE_DIR}
    )

    add_custom_command(TARGET Engine_Asset_Shaders PRE_BUILD
        COMMENT "SPIR-V --> Metal"
        COMMAND ${SPIRV_CROSS} --msl --msl-version 020101 --msl-decoration-binding --flip-vert-y --remove-unused-variables --output ${METAL_SOURCE_DIR}/${part1}.${part2}.metal ${PROJECT_BINARY_DIR}/Asset/Shaders/Vulkan/${part1}.${part2}.spv
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        DEPENDS HLSL/${part1}.${part2}.hlsl
    )
endif(APPLE)
endforeach(SHADER)

foreach(SHADER IN LISTS GLSL_SHADER_SOURCES)
    # Convert GLSL to Others
    string(REPLACE "." ";" arguments ${SHADER})
    list(GET arguments 0 part1)
    list(GET arguments 1 part2)

    add_custom_command(TARGET Engine_Asset_Shaders PRE_BUILD
        COMMENT "GLSL --> SPIR-V"
	    COMMAND ${GLSL_VALIDATOR} -V -I. -I${PROJECT_SOURCE_DIR}/Framework/Common -o ${VULKAN_SOURCE_DIR}/${part1}.${part2}.spv -e ${part1}_${part2}_main ${PROJECT_SOURCE_DIR}/Asset/Shaders/GLSL/${part1}.${part2}.glsl
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        DEPENDS GLSL/${part1}.${part2}.glsl
    )
        
endforeach(SHADER)

IF(WIN32)
    # Compile HLSL shader sources
    set(SHADER_BIN_DIR ${PROJECT_BINARY_DIR}/Asset/Shaders/HLSL)

    add_custom_command(TARGET Engine_Asset_Shaders PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BIN_DIR}
    )

    foreach(SHADER IN LISTS HLSL_SHADER_SOURCES)
        set(SHADER_BIN ${SHADER_BIN_DIR}/${SHADER}.cso)
        if (SHADER MATCHES "^([a-zA-Z0-9_]*)\.vert$")
            add_custom_command(TARGET Engine_Asset_Shaders POST_BUILD
                COMMAND ${DXC} -T vs_6_1 -E ${CMAKE_MATCH_1}_vert_main -Fo ${SHADER_BIN} -I ..\\..\\..\\ -I ..\\..\\..\\Framework\\Common ${SHADER}.hlsl
                COMMENT "Compile ${SHADER}.hlsl --> ${SHADER}.cso"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL
                DEPENDS ${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL/${SHADER}.hlsl
            )
        elseif (SHADER MATCHES "^([a-zA-Z0-9_]*)\.frag$")
            add_custom_command(TARGET Engine_Asset_Shaders POST_BUILD
                COMMAND ${DXC} -T ps_6_1 -E ${CMAKE_MATCH_1}_frag_main -Fo ${SHADER_BIN} -I ..\\..\\..\\ -I ..\\..\\..\\Framework\\Common ${SHADER}.hlsl
                COMMENT "Compile ${SHADER}.hlsl --> ${SHADER}.cso"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL
                DEPENDS ${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL/${SHADER}.hlsl
            )
        elseif (SHADER MATCHES "^([a-zA-Z0-9_]*)\.comp$")
            add_custom_command(TARGET Engine_Asset_Shaders POST_BUILD
                COMMAND ${DXC} -T cs_6_1 -E ${CMAKE_MATCH_1}_comp_main -Fo ${SHADER_BIN} -I ..\\..\\..\\ -I ..\\..\\..\\Framework\\Common ${SHADER}.hlsl
                COMMENT "Compile ${SHADER}.hlsl --> ${SHADER}.cso"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL
                DEPENDS ${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL/${SHADER}.hlsl
            )
        elseif (SHADER MATCHES "^([a-zA-Z0-9_]*)\.geom$")
            add_custom_command(TARGET Engine_Asset_Shaders POST_BUILD
                COMMAND ${DXC} -T gs_6_1 -E ${CMAKE_MATCH_1}_geom_main -Fo ${SHADER_BIN} -I ..\\..\\..\\ -I ..\\..\\..\\Framework\\Common ${SHADER}.hlsl
                COMMENT "Compile ${SHADER}.hlsl --> ${SHADER}.cso"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL
                DEPENDS ${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL/${SHADER}.hlsl
            )
        endif ()
    endforeach(SHADER)
ENDIF(WIN32)
