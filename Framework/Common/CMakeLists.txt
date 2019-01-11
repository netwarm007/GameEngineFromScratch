add_library(Common
AnimationManager.cpp
AssetLoader.cpp
BaseApplication.cpp
BlockAllocator.cpp
DebugManager.cpp
GraphicsManager.cpp
InputManager.cpp
Image.cpp
MemoryManager.cpp
Scene.cpp
SceneManager.cpp
SceneObject.cpp
StackAllocator.cpp
SceneObjectAnimation.cpp
SceneObjectMesh.cpp
SceneObjectTrack.cpp
main.cpp
)

target_link_libraries(Common
        Algorism
        DrawPass
        DrawPhase
        DispatchPass
        Geometries
        GeomMath
        ${OPENGEX_LIBRARY} 
        ${OPENDDL_LIBRARY} 
        ${XG_LIBRARY} 
        ${ZLIB_LIBRARY}
)

__add_xg_platform_dependencies(Common)

