set(shader "${SOURCE_DIR}/Test/RHI/ShadowCoordinatesTest.comp.hlsl")
set(geometry "${SOURCE_DIR}/Asset/Shaders/HLSL/shadowmap_omni.geom.hlsl")
set(spirv "${BINARY_DIR}/ShadowCoordinatesCompatibility.spv")
set(glsl "${BINARY_DIR}/ShadowCoordinatesCompatibility.comp")
set(metal "${BINARY_DIR}/ShadowCoordinatesCompatibility.metal")
set(includes "-I${SOURCE_DIR}" "-I${BINARY_DIR}" "-I${SOURCE_DIR}/Framework/Common"
             "-I${SOURCE_DIR}/Asset/Shaders/HLSL")

function(run_checked)
    execute_process(COMMAND ${ARGV} RESULT_VARIABLE result
                    OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result STREQUAL "0")
        message(FATAL_ERROR "Command failed (${result}): ${ARGV}\n${output}\n${error}")
    endif()
endfunction()

foreach(platform_macro IN ITEMS "" "OS_WINDOWS=")
    set(platform_define)
    if(NOT platform_macro STREQUAL "")
        set(platform_define --define-macro "${platform_macro}")
    endif()
    execute_process(
        COMMAND "${GLSL_VALIDATOR}" -E -S comp -D --define-macro MYGE_GLSLANG=1
                ${platform_define} ${includes} "${shader}"
        RESULT_VARIABLE result OUTPUT_VARIABLE preprocessed ERROR_VARIABLE error)
    if(NOT result STREQUAL "0")
        message(FATAL_ERROR "Shadow shader preprocessing failed: ${error}")
    endif()
    if(preprocessed MATCHES "depth_bias[ \t]*\\[[ \t]*1[ \t]*\\][ \t]*\\[[ \t]*1[ \t]*\\][ \t]*=")
        message(FATAL_ERROR "Native D3D shadow Y flip leaked into the cross-compiled shader")
    endif()

    execute_process(
        COMMAND "${GLSL_VALIDATOR}" -E -S geom -D --define-macro MYGE_GLSLANG=1
                ${platform_define} ${includes} "${geometry}"
        RESULT_VARIABLE result OUTPUT_VARIABLE preprocessed ERROR_VARIABLE error)
    if(NOT result STREQUAL "0")
        message(FATAL_ERROR "Point-shadow geometry preprocessing failed: ${error}")
    endif()
    if(NOT preprocessed MATCHES "output[ \t]*\\.[ \t]*slice[ \t]*=[ \t]*uint\\([ \t]*shadowmap_layer_index")
        message(FATAL_ERROR "Cross-compiled point shadows lost their cube-array layer offset")
    endif()
endforeach()

run_checked("${GLSL_VALIDATOR}" -V --define-macro MYGE_GLSLANG=1 --auto-map-bindings
            ${includes} -o "${spirv}" -e main "${shader}")
run_checked("${SPIRV_CROSS}" --version 430 --remove-unused-variables
            --output "${glsl}" "${spirv}")
run_checked("${GLSL_VALIDATOR}" -S comp "${glsl}")
run_checked("${SPIRV_CROSS}" --msl --msl-version 020101 --output "${metal}" "${spirv}")
foreach(stage IN ITEMS vert geom frag)
    set(stage_spirv "${BINARY_DIR}/PointShadow.${stage}.spv")
    set(stage_glsl "${BINARY_DIR}/PointShadow.${stage}")
    run_checked("${GLSL_VALIDATOR}" -V --define-macro MYGE_GLSLANG=1 ${includes}
                -o "${stage_spirv}" -e "shadowmap_omni_${stage}_main"
                "${SOURCE_DIR}/Asset/Shaders/HLSL/shadowmap_omni.${stage}.hlsl")
    run_checked("${SPIRV_CROSS}" --version 420 --remove-unused-variables
                --output "${stage_glsl}" "${stage_spirv}")
endforeach()
run_checked("${GLSL_VALIDATOR}" -l "${BINARY_DIR}/PointShadow.vert"
            "${BINARY_DIR}/PointShadow.geom" "${BINARY_DIR}/PointShadow.frag")
foreach(stage IN ITEMS vert frag)
    if(stage STREQUAL "vert")
        set(shader_name shadowmap_omni_metal)
    else()
        set(shader_name shadowmap_omni)
    endif()
    run_checked("${GLSL_VALIDATOR}" -V --define-macro MYGE_GLSLANG=1 ${includes}
                -o "${BINARY_DIR}/PointShadowMetal.${stage}.spv"
                -e "${shader_name}_${stage}_main"
                "${SOURCE_DIR}/Asset/Shaders/HLSL/${shader_name}.${stage}.hlsl")
    run_checked("${SPIRV_CROSS}" --msl --msl-version 020101 --msl-decoration-binding
                --flip-vert-y --remove-unused-variables
                --output "${BINARY_DIR}/PointShadowMetal.${stage}.metal"
                "${BINARY_DIR}/PointShadowMetal.${stage}.spv")
    if(APPLE_PLATFORM)
        run_checked(xcrun -sdk macosx metal -c "${BINARY_DIR}/PointShadowMetal.${stage}.metal"
                    -o "${BINARY_DIR}/PointShadowMetal.${stage}.air")
    endif()
endforeach()
if(APPLE_PLATFORM)
    run_checked(xcrun -sdk macosx metal -c "${metal}" -o "${BINARY_DIR}/ShadowCoordinatesCompatibility.air")
endif()
