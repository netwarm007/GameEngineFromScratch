add_custom_target(Engine_Asset_Shaders
        COMMAND echo "Start processing Engine Shaders"
    )

set(SHADER_SOURCES basic.vert basic.frag
            debug.vert debug.frag
            cubemap.frag cubemaparray.frag
            pbr.vert pbr.frag
            skybox.vert skybox.frag
            shadowmap.vert shadowmap.frag
            shadowmap_omni.vert shadowmap_omni.frag
            terrain.vert terrain.frag
            texture.frag texturearray.frag
            passthrough.vert passthrough_cube.vert
            integrateBRDF.comp
            raytrace.comp
        )

IF(WIN32)
    set(GLSL_VALIDATOR ${PROJECT_SOURCE_DIR}/External/Windows/bin/glslangValidator.exe)
    set(SPIRV_CROSS    ${PROJECT_SOURCE_DIR}/External/Windows/bin/SPIRV-Cross.exe)
ELSE(WIN32)
	set(GLSL_VALIDATOR ${PROJECT_SOURCE_DIR}/External/${MYGE_TARGET_PLATFORM}/bin/glslangValidator)
	set(SPIRV_CROSS    ${PROJECT_SOURCE_DIR}/External/${MYGE_TARGET_PLATFORM}/bin/spirv-cross)
ENDIF(WIN32)

foreach(SHADER IN LISTS SHADER_SOURCES)
    # Convert HLSL to Others
    string(REPLACE "." ";" arguments ${SHADER})
    list(GET arguments 0 part1)
    list(GET arguments 1 part2)

    set(VULKAN_SOURCE_DIR ${PROJECT_BINARY_DIR}/Asset/Shaders/Vulkan)
    set(GLSL_SOURCE_DIR ${PROJECT_BINARY_DIR}/Asset/Shaders/OpenGL)
    set(METAL_SOURCE_DIR ${PROJECT_BINARY_DIR}/Asset/Shaders/Metal)

    add_custom_command(TARGET Engine_Asset_Shaders PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${VULKAN_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${GLSL_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${METAL_SOURCE_DIR}
    )

    add_custom_command(TARGET Engine_Asset_Shaders PRE_BUILD
        COMMENT "HLSL --> SPIR-V"
	    COMMAND ${GLSL_VALIDATOR} -V -I. -I${PROJECT_SOURCE_DIR}/Framework/Common -o ${VULKAN_SOURCE_DIR}/${part1}.${part2}.spv -e ${part1}_${part2}_main ${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL/${part1}.${part2}.hlsl
        
        COMMENT "SPIR-V --> Desktop GLSL"
	    COMMAND ${SPIRV_CROSS} --version 420 --remove-unused-variables --output ${GLSL_SOURCE_DIR}/${part1}.${part2}.glsl ${PROJECT_BINARY_DIR}/Asset/Shaders/Vulkan/${part1}.${part2}.spv
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}

        COMMENT "SPIR-V --> Metal"
        COMMAND ${SPIRV_CROSS} --msl --msl-version 020101 --msl-decoration-binding --remove-unused-variables --output ${METAL_SOURCE_DIR}/${part1}.${part2}.metal ${PROJECT_BINARY_DIR}/Asset/Shaders/Vulkan/${part1}.${part2}.spv
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
endforeach(SHADER)

IF(WIN32)
    # Compile HLSL shader sources
    set(SHADER_BIN_DIR ${PROJECT_BINARY_DIR}/Asset/Shaders/HLSL)

    add_custom_command(TARGET Engine_Asset_Shaders PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BIN_DIR}
    )

    find_program(FXC fxc DOC "fx compiler"
                 PATHS
               "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Wow6432Node\\Microsoft\\Microsoft SDKs\\Windows\\v10.0;InstallationFolder]/bin/x86"
               "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Wow6432Node\\Microsoft\\Microsoft SDKs\\Windows\\v10.0;InstallationFolder]/bin/[HKEY_LOCAL_MACHINE\\SOFTWARE\\Wow6432Node\\Microsoft\\Microsoft SDKs\\Windows\\v10.0;ProductVersion].0/x86")
    if(NOT FXC)
        message(SEND_ERROR "Cannot find fxc.")
    endif(NOT FXC)

    foreach(SHADER IN LISTS SHADER_SOURCES)
        set(SHADER_BIN ${SHADER_BIN_DIR}/${SHADER}.cso)
        if (SHADER MATCHES "^([a-zA-Z0-9_]*)\.vert$")
            add_custom_command(TARGET Engine_Asset_Shaders POST_BUILD
                COMMAND ${FXC} /T vs_5_1 /E ${CMAKE_MATCH_1}_vert_main /I${PROJECT_SOURCE_DIR} /I ${PROJECT_SOURCE_DIR}/Framework/Common ${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL/${SHADER}.hlsl /Fo ${SHADER_BIN}
                COMMENT "Compile ${SHADER}.hlsl --> ${SHADER}.cso"
                DEPENDS ${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL/${SHADER}.hlsl
            )
        elseif (SHADER MATCHES "^([a-zA-Z0-9_]*)\.frag$")
            add_custom_command(TARGET Engine_Asset_Shaders POST_BUILD
                COMMAND ${FXC} /T ps_5_1 /E ${CMAKE_MATCH_1}_frag_main /I${PROJECT_SOURCE_DIR} /I ${PROJECT_SOURCE_DIR}/Framework/Common ${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL/${SHADER}.hlsl /Fo ${SHADER_BIN}
                COMMENT "Compile ${SHADER}.hlsl --> ${SHADER}.cso"
                DEPENDS ${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL/${SHADER}.hlsl
            )
        elseif (SHADER MATCHES "^([a-zA-Z0-9_]*)\.comp$")
            add_custom_command(TARGET Engine_Asset_Shaders POST_BUILD
                COMMAND ${FXC} /T cs_5_1 /E ${CMAKE_MATCH_1}_comp_main /I${PROJECT_SOURCE_DIR} /I ${PROJECT_SOURCE_DIR}/Framework/Common ${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL/${SHADER}.hlsl /Fo ${SHADER_BIN}
                COMMENT "Compile ${SHADER}.hlsl --> ${SHADER}.cso"
                DEPENDS ${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL/${SHADER}.hlsl
            )
        elseif (SHADER MATCHES "^([a-zA-Z0-9_]*)\.geom$")
            add_custom_command(TARGET Engine_Asset_Shaders POST_BUILD
                COMMAND ${FXC} /T gs_5_1 /E ${CMAKE_MATCH_1}_geom_main /I${PROJECT_SOURCE_DIR} /I ${PROJECT_SOURCE_DIR}/Framework/Common ${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL/${SHADER}.hlsl /Fo ${SHADER_BIN}
                COMMENT "Compile ${SHADER}.hlsl --> ${SHADER}.cso"
                DEPENDS ${PROJECT_SOURCE_DIR}/Asset/Shaders/HLSL/${SHADER}.hlsl
            )
        endif ()
    endforeach(SHADER)
ENDIF(WIN32)
