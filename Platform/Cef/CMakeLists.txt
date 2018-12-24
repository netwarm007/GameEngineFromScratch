find_library(CEF_LIBRARY NAMES cef libcef PATHS ${MYGE_EXTERNAL_LIBRARY_PATH} NO_CMAKE_FIND_ROOT_PATH NO_SYSTEM_ENVIRONMENT_PATH)
find_library(CEF_WRAPPER NAMES cef_dll_wrapper libcef_dll_wrapper PATHS ${MYGE_EXTERNAL_LIBRARY_PATH} NO_CMAKE_FIND_ROOT_PATH NO_SYSTEM_ENVIRONMENT_PATH)

set(PlatformInterfaceCefSources
                simple_handler.cpp
                CefApplication.cpp
)
if(${WIN32})
    list(APPEND PlatformInterfaceCefSources CefApplication.win.cpp simple_handler.win.cpp)
endif(${WIN32})
add_library(PlatformInterfaceCef ${PlatformInterfaceCefSources})
target_include_directories(PlatformInterfaceCef PRIVATE 
        "${PROJECT_SOURCE_DIR}/External/src/cef"
        "${PROJECT_SOURCE_DIR}/External/src/cef/include")
target_include_directories(PlatformInterfaceCef PUBLIC GeomMath)
target_link_libraries(PlatformInterfaceCef
    GeomMath
    ${CEF_LIBRARY}
    ${CEF_WRAPPER}
)
add_library(MyGameEngineCef CefApplicationConfig.cpp)
target_link_libraries(MyGameEngineCef 
            Common 
            EmptyRHI
            PlatformInterfaceCef
)
target_include_directories(MyGameEngineCef PRIVATE 
        "${PROJECT_SOURCE_DIR}/External/src/cef"
        "${PROJECT_SOURCE_DIR}/External/src/cef/include")