add_library(PlatformInterfaceCefCommon
                simple_handler.cpp
                CefApplication.cpp
)
target_include_directories(PlatformInterfaceCefCommon PRIVATE 
        "${PROJECT_SOURCE_DIR}/External/src/cef"
        "${PROJECT_SOURCE_DIR}/External/src/cef/include")
target_include_directories(PlatformInterfaceCefCommon PUBLIC GeomMath)
target_link_libraries(PlatformInterfaceCefCommon 
    GeomMath
)
