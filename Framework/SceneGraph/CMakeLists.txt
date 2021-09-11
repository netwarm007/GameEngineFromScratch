add_library(SceneGraph
        Scene.cpp
        SceneObject.cpp
        SceneObjectAnimation.cpp
        SceneObjectMesh.cpp
        SceneObjectTrack.cpp
        SceneObjectTexture.cpp
)

target_link_libraries(SceneGraph Common)