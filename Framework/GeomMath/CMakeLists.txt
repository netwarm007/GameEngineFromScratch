IF(${WIN32})
        SET(GEOMMATH_LIB_FILE ${CMAKE_CURRENT_BINARY_DIR}/GeomMath.lib)
        SET(ISPC_COMPILER ispc.exe)
        SET(ISPC_COMPILER_PATH ${PROJECT_SOURCE_DIR}/External/ispc/Release/)
        if($ENV{VSCMD_ARG_TGT_ARCH} MATCHES "x64")
            SET(ISPC_OPTIONS --arch=x86-64)
            SET(LIBRARIAN_COMMAND lib /MACHINE:X64 /OUT:${GEOMMATH_LIB_FILE})
        else()
            SET(ISPC_OPTIONS --arch=x86)
            SET(LIBRARIAN_COMMAND lib /MACHINE:X86 /OUT:${GEOMMATH_LIB_FILE})
        endif()
ELSE()
        SET(GEOMMATH_LIB_FILE ${CMAKE_CURRENT_BINARY_DIR}/libGeomMath.a)
        SET(ISPC_COMPILER ispc)
        SET(ISPC_COMPILER_PATH ${PROJECT_SOURCE_DIR}/External/ispc/)
        SET(ISPC_OPTIONS --arch=x86-64)
        SET(LIBRARIAN_COMMAND ar -cr -o ${GEOMMATH_LIB_FILE})
ENDIF(${WIN32})

SET(GEOMMATH_LIB_HEADER_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_custom_command(OUTPUT ${GEOMMATH_LIB_FILE}
        COMMAND ${CMAKE_COMMAND} -E env "PATH=${ISPC_COMPILER_PATH}" ${ISPC_COMPILER} ${ISPC_OPTIONS} -o CrossProduct.o -I${CMAKE_CURRENT_SOURCE_DIR} -h ${CMAKE_CURRENT_SOURCE_DIR}/include/CrossProduct.h ${CMAKE_CURRENT_SOURCE_DIR}/ispc/CrossProduct.ispc
        COMMAND ${LIBRARIAN_COMMAND} CrossProduct.o
        COMMAND rm -v CrossProduct.o
        )

add_custom_target(GeomMathLib
                   DEPENDS ${GEOMMATH_LIB_FILE}
        )

add_library(GeomMath STATIC IMPORTED GLOBAL)
add_dependencies(GeomMath GeomMathLib)

set_target_properties(GeomMath
        PROPERTIES
        IMPORTED_LOCATION ${GEOMMATH_LIB_FILE}
        INTERFACE_INCLUDE_DIRECTORIES ${GEOMMATH_LIB_HEADER_FOLDER}
        )

