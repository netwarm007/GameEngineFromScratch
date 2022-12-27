add_subdirectory(CodeGen)
add_subdirectory(Common)
add_subdirectory(DrawPass)
add_subdirectory(DrawSubPass)
add_subdirectory(DispatchPass)
add_subdirectory(GeomMath)
add_subdirectory(Manager)
add_subdirectory(Parser)
add_subdirectory(SceneGraph)

add_library(Framework INTERFACE)

if(USE_ISPC)
target_link_libraries(Framework INTERFACE
        Manager
        CodeGen
        DrawPass
        DrawSubPass
        DispatchPass
        Parser
        SceneGraph
        GeomMath
        Common
)
else()
target_link_libraries(Framework INTERFACE
        Manager
        CodeGen
        DrawPass
        DrawSubPass
        DispatchPass
        Parser
        SceneGraph
        Common
)
endif()
