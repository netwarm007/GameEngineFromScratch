function(add_platform_asset_io target source)
    add_library(${target}Bootstrap OBJECT ${source})
    add_library(${target} INTERFACE)
    # The selected bootstrap supplies AssetLoader's vtable even for scene-only consumers.
    target_link_libraries(${target} INTERFACE
        AssetIO $<TARGET_OBJECTS:${target}Bootstrap>)
endfunction()

IF(WIN32)
    # Windows specific code
    add_subdirectory(Windows)
ELSEIF(UNIX)
    IF(APPLE)
        # Apple specific code
        add_subdirectory(Darwin)
    ELSEIF(ANDROID)
        add_subdirectory(Android)
    ELSEIF(BSD)
        add_subdirectory(Linux)
	    # CEF not support BSD yet
    ELSE(APPLE)
        # Linux specific code
        add_subdirectory(Linux)
    ENDIF(APPLE)
ELSE()
    # Empty
    add_subdirectory(Empty)
ENDIF(WIN32)

if(SDL2_FOUND)
    add_subdirectory(Sdl)
endif(SDL2_FOUND)
