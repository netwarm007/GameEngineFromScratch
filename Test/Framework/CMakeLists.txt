set(FRAMEWORK_TEST_CASES 
    AnimationTest
    AssetLoaderTest 
    GeomMathTest
    GuiSubPassLayoutTest
    ImageTest
    SceneGraphTest
    SceneLoadingTest 
    SceneObjectTest
)

foreach(TEST_CASE IN LISTS FRAMEWORK_TEST_CASES)
    add_executable(${TEST_CASE} ${TEST_CASE}.cpp)
    target_link_libraries(${TEST_CASE} Framework PlatformInterface)
    add_test(NAME TEST_${TEST_CASE} COMMAND ${TEST_CASE})
endforeach()

target_link_libraries(SceneGraphTest ImageDecoders)

add_executable(SceneObjectTextureTest SceneObjectTextureTest.cpp)
target_link_libraries(SceneObjectTextureTest PRIVATE
    SceneResources AssetIO PlatformAssetIO)
add_test(NAME TEST_SceneObjectTextureTest COMMAND SceneObjectTextureTest)

add_executable(SceneServicesTest SceneServicesTest.cpp)
target_link_libraries(SceneServicesTest PRIVATE SceneServices)
add_test(NAME TEST_SceneServicesTest COMMAND SceneServicesTest)

add_executable(SceneCameraTest SceneCameraTest.cpp)
target_link_libraries(SceneCameraTest PRIVATE SceneCamera)
add_test(NAME TEST_SceneCameraTest COMMAND SceneCameraTest)

add_executable(SceneLightingTest SceneLightingTest.cpp)
target_link_libraries(SceneLightingTest PRIVATE SceneLighting)
add_test(NAME TEST_SceneLightingTest COMMAND SceneLightingTest)

add_executable(RenderServicesTest RenderServicesTest.cpp)
target_link_libraries(RenderServicesTest PRIVATE
    RenderServices PlatformAssetIO ImageDecoders)
add_test(NAME TEST_RenderServicesTest COMMAND RenderServicesTest)

add_executable(ApplicationLifetimeTest ApplicationLifetimeTest.cpp)
target_link_libraries(ApplicationLifetimeTest PRIVATE Manager)
add_test(NAME TEST_ApplicationLifetimeTest COMMAND ApplicationLifetimeTest)

add_executable(PhysicsSceneIntegrationTest PhysicsSceneIntegrationTest.cpp)
target_link_libraries(PhysicsSceneIntegrationTest PRIVATE
    BulletPhysics MyPhysics Parser AssetIO PlatformAssetIO SplashSequence)
add_test(NAME TEST_PhysicsSceneIntegrationTest COMMAND PhysicsSceneIntegrationTest)
