SET(GEOMMATH_LIB_FILE ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}GeomMath${CMAKE_STATIC_LIBRARY_SUFFIX})
IF(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
    SET(ISPC_COMPILER ispc.exe)
    SET(ISPC_COMPILER_PATH ${PROJECT_SOURCE_DIR}/External/ispc/Release/)
    IF(NOT ${CMAKE_CROSSCOMPILING})
            if(${ARCH_BIT} MATCHES 64)
            SET(ISPC_OPTIONS --arch=x86-64 --target=host)
            SET(CMAKE_AR lib)
            SET(ISPC_LIBRARIAN_OPTIONS /MACHINE:X64 /OUT:${GEOMMATH_LIB_FILE})
        else()
            SET(ISPC_OPTIONS --arch=x86 --target=host)
            SET(CMAKE_AR lib)
            SET(ISPC_LIBRARIAN_OPTIONS /MACHINE:X86 /OUT:${GEOMMATH_LIB_FILE})
        endif()
    ELSE()
            SET(ISPC_OPTIONS --arch=arm --cpu=cortex-a9 --target=neon-i32x4)
            SET(CMAKE_AR ${SNC_TOOLCHAIN_DIR}psp2snarl -rcv ${GEOMMATH_LIB_FILE})
    ENDIF()
ELSEIF(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
    SET(ISPC_COMPILER ispc)
    SET(ISPC_COMPILER_PATH ${PROJECT_SOURCE_DIR}/External/ispc/)
    SET(ISPC_OPTIONS --arch=x86-64)
    SET(ISPC_LIBRARIAN_OPTIONS -rcv -o ${GEOMMATH_LIB_FILE})
ENDIF()

SET(GEOMMATH_LIB_HEADER_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(GeomMath STATIC IMPORTED GLOBAL)
add_dependencies(GeomMath GeomMathLib)

set_target_properties(GeomMath
        PROPERTIES
        IMPORTED_LOCATION ${GEOMMATH_LIB_FILE}
        INTERFACE_INCLUDE_DIRECTORIES ${GEOMMATH_LIB_HEADER_FOLDER}
        )

add_subdirectory(ispc)
add_subdirectory(test)

