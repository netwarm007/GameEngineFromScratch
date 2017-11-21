# Empty
IF(${CMAKE_SYSTEM_NAME} MATCHES Orbis)
ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES PSP2)
ELSE()
    add_subdirectory(Empty)
    add_subdirectory(OpenGL)
    IF(${WIN32})
        # Windows specific code
        add_subdirectory(D2d)
        add_subdirectory(D3d)
    ENDIF(${WIN32})
ENDIF()
