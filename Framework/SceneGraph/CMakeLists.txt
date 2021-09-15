add_library(SceneGraph
        Scene.cpp
        SceneObject.cpp
        SceneObjectAnimation.cpp
        SceneObjectMesh.cpp
        SceneObjectTrack.cpp
        SceneObjectTexture.cpp
)

target_link_libraries(SceneGraph
        ${XG_LIBRARY} 
        ${ZLIB_LIBRARY}
)

__add_xg_platform_dependencies(SceneGraph)