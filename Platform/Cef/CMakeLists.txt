if(${APPLE})
find_library(COCOA_LIBRARY Cocoa required)
find_library(CEF_LIBRARY NAMES "Chromium Embedded Framework" PATHS ${MYGE_EXTERNAL_FRAMEWORK_PATH} NO_CMAKE_FIND_ROOT_PATH NO_SYSTEM_ENVIRONMENT_PATH)
else(${APPLIE})
find_library(CEF_LIBRARY NAMES cef libcef PATHS ${MYGE_EXTERNAL_LIBRARY_PATH} NO_CMAKE_FIND_ROOT_PATH NO_SYSTEM_ENVIRONMENT_PATH)
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
    target_link_libraries(PlatformInterfaceCef
        GeomMath
        ${CEF_WRAPPER}
        ${CEF_LIBRARY}
    )
endif(${APPLE})

add_library(MyGameEngineCef CefApplicationConfig.cpp)
target_link_libraries(MyGameEngineCef 
            Common 
            EmptyRHI
            PlatformInterfaceCef
)
