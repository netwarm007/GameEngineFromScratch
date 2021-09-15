add_subdirectory(Algorism)
add_subdirectory(Common)
add_subdirectory(DrawPass)
add_subdirectory(DrawSubPass)
add_subdirectory(DispatchPass)
add_subdirectory(GeomMath)
add_subdirectory(Geometries)
add_subdirectory(Manager)
add_subdirectory(Parser)
add_subdirectory(SceneGraph)

add_library(Framework INTERFACE)

target_link_libraries(Framework INTERFACE
        Manager
        Algorism
        DrawPass
        DrawSubPass
        DispatchPass
        Parser
        Geometries
        SceneGraph
        GeomMath
        Common
)
