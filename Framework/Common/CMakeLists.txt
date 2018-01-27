add_library(Common
Allocator.cpp
AssetLoader.cpp
BaseApplication.cpp
GameLogic.cpp
GraphicsManager.cpp
InputManager.cpp
Image.cpp
MemoryManager.cpp
PhysicsManager.cpp
Scene.cpp
SceneManager.cpp
SceneObject.cpp
main.cpp
)
target_link_libraries(Common 
        GeomMath
        ${OPENGEX_LIBRARY} 
        ${OPENDDL_LIBRARY} 
        ${XG_LIBRARY} 
        ${ZLIB_LIBRARY}
        ${BULLET_DYNAMICS_LIBRARY} 
        ${BULLET_COLLISION_LIBRARY} 
        ${BULLET_LINEARMATH_LIBRARY}
)

__add_xg_platform_dependencies(Common)

