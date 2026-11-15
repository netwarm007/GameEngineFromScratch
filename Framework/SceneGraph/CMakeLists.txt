add_library(SceneGraph
        Scene.cpp
        SceneObject.cpp
        SceneObjectAnimation.cpp
        SceneObjectMesh.cpp
        SceneObjectTrack.cpp
)

target_link_libraries(SceneGraph ${XG_LIBRARY})

__add_xg_platform_dependencies(SceneGraph)
target_link_libraries(SceneGraph Common)