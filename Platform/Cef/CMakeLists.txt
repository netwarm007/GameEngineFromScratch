if(${APPLE})
    find_library(CEF_LIBRARY NAMES "Chromium Embedded Framework" PATHS ${MYGE_EXTERNAL_FRAMEWORK_PATH} NO_CMAKE_FIND_ROOT_PATH NO_SYSTEM_ENVIRONMENT_PATH)
else(${APPLIE})
    find_library(CEF_LIBRARY NAMES cef libcef PATHS ${MYGE_EXTERNAL_LIBRARY_PATH} NO_CMAKE_FIND_ROOT_PATH NO_SYSTEM_ENVIRONMENT_PATH)
endif(${APPLE})

find_library(CEF_WRAPPER NAMES cef_dll_wrapper libcef_dll_wrapper PATHS ${MYGE_EXTERNAL_LIBRARY_PATH} NO_CMAKE_FIND_ROOT_PATH NO_SYSTEM_ENVIRONMENT_PATH)

set(CefSources
                simple_handler.cpp
                CefApplication.cpp
)

IF(WIN32)
list(APPEND CefSources
                simple_handler.win.cpp
                CefApplication.win.cpp
)
ELSEIF(APPLE)
list(APPEND CefSources
                simple_handler.mac.mm
                CefApplication.mac.mm
)
ELSEIF(UNIX)
list(APPEND CefSources
                simple_handler.nix.cpp
                CefApplication.nix.cpp
)
ENDIF(WIN32)

add_library(CEF ${CefSources})

target_link_libraries(CEF
    ${CEF_WRAPPER}
    ${CEF_LIBRARY}
)
