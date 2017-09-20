set(FUNCTIONS CrossProduct DotProduct MulByElement Transpose)
add_custom_command(OUTPUT ${GEOMMATH_LIB_FILE}
        COMMAND echo ${FUNCTIONS}
        )
foreach(FUNC IN LISTS FUNCTIONS)
    add_custom_command(OUTPUT ${GEOMMATH_LIB_FILE} APPEND
        COMMAND ${CMAKE_COMMAND} -E env "PATH=${ISPC_COMPILER_PATH}" ${ISPC_COMPILER} ${ISPC_OPTIONS} -o ${FUNC}.o -I${CMAKE_CURRENT_SOURCE_DIR} -h ${GEOMMATH_LIB_HEADER_FOLDER}/${FUNC}.h --target=host ${CMAKE_CURRENT_SOURCE_DIR}/${FUNC}.ispc
        COMMAND ${LIBRARIAN_COMMAND} ${FUNC}.o
        COMMAND rm -v ${FUNC}.o
        DEPENDS ${FUNC}.ispc
        MAIN_DEPENDENCY ${FUNC}.ispc
        COMMENT "Building ${FUNC}"
        )
endforeach(FUNC)

add_custom_target(GeomMathLib
                   DEPENDS ${GEOMMATH_LIB_FILE}
        )

