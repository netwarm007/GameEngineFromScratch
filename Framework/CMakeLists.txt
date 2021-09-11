add_subdirectory(Algorism)
add_subdirectory(Common)
add_subdirectory(DrawPass)
add_subdirectory(DrawSubPass)
add_subdirectory(DispatchPass)
add_subdirectory(GeomMath)
add_subdirectory(Geometries)
add_subdirectory(Parser)
add_subdirectory(SceneGraph)

add_library(Framework INTERFACE)

target_link_libraries(Framework INTERFACE
        Algorism
        DrawPass
        DrawSubPass
        DispatchPass
        Geometries
        Parser
        GeomMath
        SceneGraph
)
