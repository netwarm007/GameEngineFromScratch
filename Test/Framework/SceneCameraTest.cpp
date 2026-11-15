#include <cmath>
#include <iostream>
#include <limits>
#include <string>

#include "Scene.hpp"
#include "SceneCamera.hpp"

using namespace My;

namespace {
int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

bool Near(float actual, float expected) {
    return std::abs(actual - expected) < 0.0001f;
}

float ProjectDepth(const CameraFrame& camera, float viewZ) {
    Vector4f point{0.0f, 0.0f, viewZ, 1.0f};
    Transform(point, camera.projectionMatrix);
    return point[2] / point[3];
}

void ExpectView(const CameraFrame& camera, const Vector3f& eye,
                const Vector3f& target) {
    Vector4f eyeInView{eye[0], eye[1], eye[2], 1.0f};
    Transform(eyeInView, camera.viewMatrix);
    Expect(
        Near(eyeInView[0], 0) && Near(eyeInView[1], 0) && Near(eyeInView[2], 0),
        "camera world position maps to the view origin");
    Vector4f targetInView{target[0], target[1], target[2], 1.0f};
    Transform(targetInView, camera.viewMatrix);
    Expect(Near(targetInView[0], 0) && Near(targetInView[1], 0) &&
               targetInView[2] < 0,
           "target lies on the right-handed negative view Z axis");
    Expect(Near(camera.position[0], eye[0]) &&
               Near(camera.position[1], eye[1]) &&
               Near(camera.position[2], eye[2]) && camera.position[3] == 0,
           "published position matches the view with the existing shader W");
}

std::shared_ptr<SceneCameraNode> AddCamera(Scene& scene) {
    auto node = std::make_shared<SceneCameraNode>("camera node");
    node->AddSceneObjectRef("camera");
    node->AppendTransform("position", std::make_shared<SceneObjectTranslation>(
                                          3.0f, -8.0f, 2.0f));
    node->MoveBy(1.0f, 0.0f, 1.0f);
    node->SetTarget({1.0f, 2.0f, 0.0f});
    scene.SceneGraph->AppendChild(node);
    scene.CameraNodes.emplace("camera", node);
    auto camera = std::make_shared<SceneObjectPerspectiveCamera>(PI / 2.0f);
    std::string nearParam = "near", farParam = "far";
    camera->SetParam(nearParam, 0.5f);
    camera->SetParam(farParam, 75.0f);
    scene.Cameras.emplace("camera", camera);
    return node;
}

void TestSceneCameraAndResize() {
    Scene scene("perspective");
    auto node = AddCamera(scene);
    CameraFrame camera;
    for (bool openGL : {false, true}) {
        Expect(PrepareSceneCamera(scene, 800, 400, openGL, camera),
               "prepare scene camera without an application or GPU");
        ExpectView(camera, {4.0f, -8.0f, 3.0f}, node->GetTarget());
        Expect(
            Near(ProjectDepth(camera, -0.5f), openGL ? -1.0f : 0.0f) &&
                Near(ProjectDepth(camera, -75.0f), 1.0f),
            "camera clipping planes retain the selected backend depth range");
        Expect(Near(camera.projectionMatrix[0][0], 0.5f) &&
                   Near(camera.projectionMatrix[1][1], 1.0f),
               "scene FOV and canvas aspect determine projection scaling");
        Expect(PrepareSceneCamera(scene, 400, 800, openGL, camera),
               "prepare portrait render canvas");
        Expect(Near(camera.projectionMatrix[0][0], 2.0f) &&
                   Near(camera.projectionMatrix[1][1], 1.0f),
               "canvas resizing changes horizontal scale, not vertical FOV");
    }
}

void TestSceneReplacementAndDefaultCamera() {
    Scene first("camera scene");
    auto node = AddCamera(first);
    CameraFrame camera;
    Expect(PrepareSceneCamera(first, 640, 480, false, camera),
           "prepare first scene");
    node->MoveBy(2.0f, 1.0f, 0.0f);
    Expect(PrepareSceneCamera(first, 640, 480, false, camera),
           "read updated camera transform each frame");
    ExpectView(camera, {6.0f, -7.0f, 3.0f}, node->GetTarget());

    Scene replacement("no camera");
    for (bool openGL : {false, true}) {
        Expect(PrepareSceneCamera(replacement, 640, 480, openGL, camera),
               "prepare built-in camera for replacement scene");
        ExpectView(camera, {0.0f, -5.0f, 0.0f}, {0.0f, 0.0f, 0.0f});
        Expect(Near(ProjectDepth(camera, -10.0f), openGL ? -1.0f : 0.0f) &&
                   Near(ProjectDepth(camera, -100.0f), 1.0f),
               "built-in camera retains its existing clipping planes");
    }
}

void TestFailurePreservesOutput() {
    Scene scene("invalid camera");
    auto node = AddCamera(scene);
    CameraFrame camera;
    Expect(PrepareSceneCamera(scene, 640, 480, false, camera),
           "prepare valid output before failure");
    const CameraFrame previous = camera;
    auto expectUnchanged = [&] {
        for (int i = 0; i < 4; ++i) {
            Expect(camera.position[i] == previous.position[i],
                   "failed preparation preserves camera position");
            for (int j = 0; j < 4; ++j) {
                Expect(camera.viewMatrix[i][j] == previous.viewMatrix[i][j] &&
                           camera.projectionMatrix[i][j] ==
                               previous.projectionMatrix[i][j],
                       "failed preparation preserves both matrices");
            }
        }
    };
    Expect(!PrepareSceneCamera(scene, 0, 480, false, camera) &&
               !PrepareSceneCamera(scene, 640, 0, false, camera),
           "reject zero canvas dimensions");
    expectUnchanged();
    scene.Cameras.clear();
    Expect(!PrepareSceneCamera(scene, 640, 480, false, camera),
           "reject unresolved camera reference instead of dereferencing null");
    expectUnchanged();
    auto perspective = std::make_shared<SceneObjectPerspectiveCamera>(
        std::numeric_limits<float>::quiet_NaN());
    scene.Cameras.emplace("camera", perspective);
    Expect(!PrepareSceneCamera(scene, 640, 480, false, camera),
           "reject non-finite FOV");
    expectUnchanged();
    std::string fov = "fov", nearParam = "near", farParam = "far";
    perspective->SetParam(fov, PI / 2.0f);
    perspective->SetParam(nearParam, 20.0f);
    perspective->SetParam(farParam, 10.0f);
    Expect(!PrepareSceneCamera(scene, 640, 480, false, camera),
           "reject reversed clipping planes");
    expectUnchanged();
}
}  // namespace

int main() {
    TestSceneCameraAndResize();
    TestSceneReplacementAndDefaultCamera();
    TestFailurePreservesOutput();
    return failures == 0 ? 0 : 1;
}
