set(FUNCTIONS CrossProduct MulByElement Transpose Normalize
              Transform AddByElement SubByElement MatrixUtil
              InverseMatrix DCT Absolute Pow DivByElement 
        )

foreach(FUNC IN LISTS FUNCTIONS)
    IF(APPLE)
        add_custom_command(OUTPUT ${FUNC}.o
            COMMAND ${CMAKE_COMMAND} -E env "PATH=${ISPC_COMPILER_PATH}" ${ISPC_COMPILER} ${ISPC_OPTIONS} -o ${FUNC}.o ${CMAKE_CURRENT_SOURCE_DIR}/${FUNC}.ispc
            MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/${FUNC}.ispc
            COMMENT "Building ${FUNC}"
            )
    ELSEIF(${CMAKE_HOST_SYSTEM_NAME} MATCHES FreeBSD)
        add_custom_command(OUTPUT ${FUNC}.o
            COMMAND ${CMAKE_COMMAND} -E env "PATH=${ISPC_COMPILER_PATH}" ${ISPC_COMPILER} ${ISPC_OPTIONS} -o ${FUNC}.o ${CMAKE_CURRENT_SOURCE_DIR}/${FUNC}.ispc
            MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/${FUNC}.ispc
            COMMENT "Building ${FUNC}"
            )
    ELSEIF(${CMAKE_HOST_SYSTEM_NAME} MATCHES Windows)
        add_custom_command(OUTPUT ${FUNC}.o
            COMMAND ${CMAKE_COMMAND} -E env "PATH=${ISPC_COMPILER_PATH}" ${ISPC_COMPILER} ${ISPC_OPTIONS} -o ${FUNC}.o ${CMAKE_CURRENT_SOURCE_DIR}/${FUNC}.ispc
            MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/${FUNC}.ispc
            COMMENT "Building ${FUNC}"
            )
    ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES Android)
        add_custom_command(OUTPUT ${FUNC}.o
            COMMAND ${CMAKE_COMMAND} -E env "PATH=${ISPC_COMPILER_PATH}" ${ISPC_COMPILER} ${ISPC_OPTIONS} --emit-llvm -o ${FUNC}.bc ${CMAKE_CURRENT_SOURCE_DIR}/${FUNC}.ispc
            COMMAND ${CMAKE_COMMAND} -E env "PATH=${LLVM_COMPILER_PATH}" llc -O3 -float-abi=soft -march=arm -mcpu=cortex-a9 -thread-model=posix -filetype=obj -o=${FUNC}.o ${FUNC}.bc
            MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/${FUNC}.ispc
            COMMENT "Building ${FUNC}"
            )
    ELSE()
        add_custom_command(OUTPUT ${FUNC}.o
            COMMAND ${CMAKE_COMMAND} -E env "PATH=${ISPC_COMPILER_PATH}" ${ISPC_COMPILER} ${ISPC_OPTIONS} -o ${FUNC}.o ${CMAKE_CURRENT_SOURCE_DIR}/${FUNC}.ispc
            MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/${FUNC}.ispc
            COMMENT "Building ${FUNC}"
            )
    ENDIF()

    list(APPEND OBJECTS ${FUNC}.o)

endforeach(FUNC)

IF(${WIN32})
add_custom_command(OUTPUT ${GEOMMATH_LIB_FILE}
        COMMAND ${CMAKE_AR} ${ISPC_LIBRARIAN_OPTIONS} ${OBJECTS}
        DEPENDS ${OBJECTS}
        )
ELSE(${WIN32})
add_custom_command(OUTPUT ${GEOMMATH_LIB_FILE}
        COMMAND ${CMAKE_AR} ${ISPC_LIBRARIAN_OPTIONS} ${OBJECTS}
        COMMAND ${CMAKE_RANLIB} ${GEOMMATH_LIB_FILE}
        DEPENDS ${OBJECTS}
        )
ENDIF(${WIN32})

add_custom_target(IspcGeomMathLib
                   DEPENDS ${GEOMMATH_LIB_FILE}
        )

