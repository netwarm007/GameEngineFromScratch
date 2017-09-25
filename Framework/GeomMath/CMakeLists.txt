IF(${WIN32})
        SET(GEOMMATH_LIB_FILE ${CMAKE_CURRENT_BINARY_DIR}/GeomMath.lib)
        SET(ISPC_COMPILER ispc.exe)
        SET(ISPC_OPTIONS --arch=x86-64 --target=host)
        SET(LIBRARIAN_COMMAND lib)
        SET(LIBRARIAN_OPTIONS /OUT:${GEOMMATH_LIB_FILE})
ELSE(${WIN32})
        SET(GEOMMATH_LIB_FILE ${CMAKE_CURRENT_BINARY_DIR}/libGeomMath.a)
        SET(ISPC_COMPILER ispc)
        SET(ISPC_OPTIONS --arch=x86-64 --target=host)
        SET(LIBRARIAN_COMMAND ar)
        SET(LIBRARIAN_OPTIONS -cr -o ${GEOMMATH_LIB_FILE})
ENDIF(${WIN32})

SET(GEOMMATH_LIB_HEADER_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_custom_command(OUTPUT ${GEOMMATH_LIB_FILE}
        COMMAND ${ISPC_COMPILER} ${ISPC_OPTIONS} -o CrossProduct.o -I${CMAKE_CURRENT_SOURCE_DIR} -h ${CMAKE_CURRENT_SOURCE_DIR}/include/CrossProduct.h ${CMAKE_CURRENT_SOURCE_DIR}/ispc/CrossProduct.ispc
        COMMAND ${LIBRARIAN_COMMAND} ${LIBRARIAN_OPTIONS} CrossProduct.o
        )

add_custom_target(ISPC
                   DEPENDS ${GEOMMATH_LIB_FILE}
        )

add_library(GeomMath STATIC IMPORTED GLOBAL)
add_dependencies(GeomMath ISPC)

set_target_properties(GeomMath
        PROPERTIES
        IMPORTED_LOCATION ${GEOMMATH_LIB_FILE}
        INTERFACE_INCLUDE_DIRECTORIES ${GEOMMATH_LIB_HEADER_FOLDER}
        )

set_directory_properties(
        PROPERTIES
        ADDITIONAL_MAKE_CLEAN_FILES CrossProduct.o
        )

