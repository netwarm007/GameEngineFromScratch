IF(${WIN32})
SET(GEOMMATH_LIB_FILE ${CMAKE_CURRENT_SOURCE_DIR}/GeomMath.lib)
ELSE(${WIN32})
SET(GEOMMATH_LIB_FILE ${CMAKE_CURRENT_SOURCE_DIR}/libGeomMath.a)
ENDIF(${WIN32})

SET(GEOMMATH_LIB_HEADER_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_custom_command(OUTPUT ${GEOMMATH_LIB_FILE}
           COMMAND ${PROJECT_SOURCE_DIR}/External/ispc/ispc -o CrossProduct.o -I${CMAKE_CURRENT_SOURCE_DIR} -h ${CMAKE_CURRENT_SOURCE_DIR}/include/CrossProduct.h ${CMAKE_CURRENT_SOURCE_DIR}/ispc/CrossProduct.ispc
           COMMAND ${CMAKE_AR} cr ${GEOMMATH_LIB_FILE} CrossProduct.o
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

