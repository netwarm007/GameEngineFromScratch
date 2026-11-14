add_library(SceneGraph
        Scene.cpp
        SceneObject.cpp
        SceneObjectAnimation.cpp
        SceneObjectMesh.cpp
        SceneObjectTrack.cpp
        SceneObjectTexture.cpp
)

if(MSVC)
    target_link_libraries(SceneGraph
            ${XG_LIBRARY}
            optimized ${ZLIB_LIBRARY}
            debug ${ZLIB_LIBRARY_DEBUG}
    )
else()
    target_link_libraries(SceneGraph
            ${XG_LIBRARY}
            ${ZLIB_LIBRARY}
    )
endif()

__add_xg_platform_dependencies(SceneGraph)
target_link_libraries(SceneGraph AssetIO Common)