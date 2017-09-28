set(FUNCTIONS CrossProduct DotProduct MulByElement Transpose Normalize
              Transform AddByElement SubByElement
        )

foreach(FUNC IN LISTS FUNCTIONS)
    add_custom_command(OUTPUT ${FUNC}.o
        COMMAND ${CMAKE_COMMAND} -E env "PATH=${ISPC_COMPILER_PATH}" ${ISPC_COMPILER} ${ISPC_OPTIONS} -o ${FUNC}.o -h ${GEOMMATH_LIB_HEADER_FOLDER}/${FUNC}.h ${CMAKE_CURRENT_SOURCE_DIR}/${FUNC}.ispc
        MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/${FUNC}.ispc
        COMMENT "Building ${FUNC}"
        )

    list(APPEND OBJECTS ${FUNC}.o)

endforeach(FUNC)

add_custom_command(OUTPUT ${GEOMMATH_LIB_FILE}
        COMMAND ${CMAKE_AR} ${ISPC_LIBRARIAN_OPTIONS} ${OBJECTS}
        COMMAND rm -v ${OBJECTS}
        DEPENDS ${OBJECTS}
        )

add_custom_target(GeomMathLib
                   DEPENDS ${GEOMMATH_LIB_FILE}
        )

