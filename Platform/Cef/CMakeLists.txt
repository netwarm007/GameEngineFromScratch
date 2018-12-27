if(${APPLE})
    find_library(COCOA_LIBRARY Cocoa required)
    find_library(CEF_LIBRARY NAMES "Chromium Embedded Framework" PATHS ${MYGE_EXTERNAL_FRAMEWORK_PATH} NO_CMAKE_FIND_ROOT_PATH NO_SYSTEM_ENVIRONMENT_PATH)
else(${APPLIE})
    if(${WIN32})
        find_library(CEF_LIBRARY NAMES cef libcef PATHS ${MYGE_EXTERNAL_LIBRARY_PATH} NO_CMAKE_FIND_ROOT_PATH NO_SYSTEM_ENVIRONMENT_PATH)
    else(${WIN32})
        find_library(CEF_LIBRARY NAMES cef libcef PATHS ${MYGE_EXTERNAL_LIBRARY_PATH} NO_CMAKE_FIND_ROOT_PATH NO_SYSTEM_ENVIRONMENT_PATH)
        find_library(X11_LIBRARY X11)
    endif(${WIN32})
endif(${APPLE})
find_library(CEF_WRAPPER NAMES cef_dll_wrapper libcef_dll_wrapper PATHS ${MYGE_EXTERNAL_LIBRARY_PATH} NO_CMAKE_FIND_ROOT_PATH NO_SYSTEM_ENVIRONMENT_PATH)

set(PlatformInterfaceCefSources
                simple_handler.cpp
                CefApplication.cpp
)
if(${WIN32})
    list(APPEND PlatformInterfaceCefSources CefApplication.win.cpp simple_handler.win.cpp)
endif(${WIN32})
if(${APPLE})
    list(APPEND PlatformInterfaceCefSources CefApplication.mac.mm simple_handler.mac.mm)
else(${APPLE})
    if(${UNIX})
        list(APPEND PlatformInterfaceCefSources CefApplication.nix.cpp simple_handler.nix.cpp)
endif(${UNIX})
endif(${APPLE})
add_library(PlatformInterfaceCef ${PlatformInterfaceCefSources})
target_include_directories(PlatformInterfaceCef PUBLIC GeomMath)

if(${APPLE})
    target_link_libraries(PlatformInterfaceCef
        GeomMath
        ${CEF_WRAPPER}
        ${CEF_LIBRARY}
        ${COCOA_LIBRARY}
    )
else(${APPLE})
    if(${WIN32})
        target_link_libraries(PlatformInterfaceCef
            GeomMath
            ${CEF_WRAPPER}
            ${CEF_LIBRARY}
        )
    else(${WIN32})
        target_link_libraries(PlatformInterfaceCef
            GeomMath
            ${CEF_WRAPPER}
            ${CEF_LIBRARY}
            ${X11_LIBRARY}
        )
    endif(${WIN32})
endif(${APPLE})

add_library(MyGameEngineCef CefApplicationConfig.cpp)
target_link_libraries(MyGameEngineCef 
            Common 
            EmptyRHI
            PlatformInterfaceCef
)
