#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "AssetLoader.hpp"
#include "Bullet/BulletPhysicsManager.hpp"
#include "ISceneManager.hpp"
#include "My/MyPhysicsManager.hpp"
#include "OGEX.hpp"

using namespace My;

namespace {
int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

bool Near(float lhs, float rhs) { return std::abs(lhs - rhs) < 0.001f; }

class TestScenes : public ISceneManager {
   public:
    int Initialize() override { return 0; }
    void Finalize() override {}
    void Tick() override {}
    int LoadScene(const char*) override { return -1; }
    uint64_t GetSceneRevision() const override { return revision; }
    const std::shared_ptr<Scene> GetSceneForRendering() const override {
        return scene;
    }
    const std::shared_ptr<Scene> GetSceneForPhysicalSimulation() const override {
        return scene;
    }
    void ResetScene() override { ++revision; }
    std::weak_ptr<BaseSceneNode> GetRootNode() const override { return {}; }
    std::weak_ptr<SceneGeometryNode> GetSceneGeometryNode(
        const std::string&) const override { return {}; }
    std::weak_ptr<SceneObjectGeometry> GetSceneGeometryObject(
        const std::string&) const override { return {}; }

    std::shared_ptr<Scene> scene;
    uint64_t revision = 0;
};

class SteppingBullet : public BulletPhysicsManager {
   public:
    using BulletPhysicsManager::BulletPhysicsManager;
    void Step() {
        m_btDynamicsWorld->stepSimulation(1.0f / 60.0f, 10);
        PublishTransforms();
    }

    int BodyCount() const { return m_btDynamicsWorld->getNumCollisionObjects(); }

    void SetVelocity(SceneGeometryNode& node, const Vector3f& velocity) {
        auto* body = m_RigidBodies.at(&node).body;
        body->setLinearVelocity(
            btVector3(velocity[0], velocity[1], velocity[2]));
        body->activate(true);
    }
};

template <class Physics>
void TestBindingsAndTransforms() {
    TestScenes scenes;
    Physics physics(scenes);
    Expect(physics.Initialize() == 0, "physics initializes without application");
    auto node = std::make_shared<SceneGeometryNode>("body");
    node->MoveBy(2.0f, 3.0f, 10.0f);
    SceneObjectGeometry geometry;
    geometry.SetCollisionType(SceneObjectCollisionType::kSceneObjectCollisionTypeSphere);
    const float radius[] = {1.0f};
    geometry.SetCollisionParameters(radius, 1);
    Expect(!physics.UpdateRigidBodyTransform(*node),
           "unbound transform update reports no body");
    Expect(!physics.ApplyCentralForce(*node, Vector3f({1.0f, 0.0f, 0.0f})),
           "unbound force reports no body");
    physics.CreateRigidBody(node, geometry);
    Expect(Near(node->GetRenderTransform()[3][2], 10.0f),
           "body creation publishes initial transform");

    node->MoveBy(1.0f, 0.0f, 0.0f);
    Expect(Near(node->GetRenderTransform()[3][0], 2.0f),
           "rendering uses published simulation rather than authored transform");
    Expect(physics.UpdateRigidBodyTransform(*node),
           "bound transform update succeeds");
    Expect(Near(node->GetRenderTransform()[3][0], 3.0f),
           "explicit transform update republishes body position");
    physics.DeleteRigidBody(*node);
    node->MoveBy(1.0f, 0.0f, 0.0f);
    Expect(Near(node->GetRenderTransform()[3][0], 4.0f),
           "deletion restores authored/runtime transform rendering");
    physics.DeleteRigidBody(*node);

    physics.CreateRigidBody(node, geometry);
    physics.CreateRigidBody(node, geometry);
    std::weak_ptr<SceneGeometryNode> retained = node;
    node.reset();
    Expect(!retained.expired(), "binding retains node until body cleanup");
    physics.ClearRigidBodies();
    Expect(retained.expired(), "clearing bindings releases old scene nodes");
    physics.ClearRigidBodies();

    auto unsupported = std::make_shared<SceneGeometryNode>("no collision");
    SceneObjectGeometry noCollision;
    physics.CreateRigidBody(unsupported, noCollision);
    unsupported->MoveBy(0.0f, 0.0f, 7.0f);
    Expect(Near(unsupported->GetRenderTransform()[3][2], 7.0f) &&
               !physics.UpdateRigidBodyTransform(*unsupported),
           "non-collidable geometry keeps normal scene transforms");
    physics.Finalize();
}

void TestBulletSimulationPublication() {
    TestScenes scenes;
    SteppingBullet physics(scenes);
    Expect(physics.Initialize() == 0, "Bullet initializes");
    auto node = std::make_shared<SceneGeometryNode>("falling");
    node->MoveBy(0.0f, 0.0f, 10.0f);
    SceneObjectGeometry geometry;
    geometry.SetCollisionType(SceneObjectCollisionType::kSceneObjectCollisionTypeSphere);
    const float radius[] = {1.0f};
    geometry.SetCollisionParameters(radius, 1);
    physics.CreateRigidBody(node, geometry);
    physics.CreateRigidBody(node, geometry);
    Expect(physics.BodyCount() == 1, "replacing binding removes previous body");
    Expect(physics.ApplyCentralForce(*node, Vector3f({100.0f, 0.0f, 0.0f})),
           "force resolves backend body through scene node");
    for (int i = 0; i < 30; ++i) {
        physics.Step();
    }
    auto transform = node->GetRenderTransform();
    Expect(transform[3][2] < 9.5f && transform[3][0] > 0.1f,
           "renderer-facing transform follows gravity and applied force");
    Expect(Near((*node->GetCalculatedTransform())[3][2], 10.0f),
           "simulation does not overwrite authored scene transforms");
    physics.ClearRigidBodies();
    Expect(physics.BodyCount() == 0 &&
               Near(node->GetRenderTransform()[3][2], 10.0f),
           "world cleanup removes bodies and simulation overrides");
    physics.Finalize();
}

void TestBilliardScene() {
    AssetLoader assets;
    Expect(assets.Initialize() == 0, "Billiard assets initialize");
    OgexParser parser;
    auto scene = parser.Parse(
        assets.SyncOpenAndReadTextFileToString("Scene/billiard.ogex"));
    Expect(scene != nullptr, "actual Billiard scene parses");
    if (!scene) {
        assets.Finalize();
        return;
    }
    auto cue = scene->LUT_Name_GeometryNode.at("pbb_cue").lock();
    Expect(cue != nullptr, "Billiard cue ball exists");
    if (!cue) {
        assets.Finalize();
        return;
    }
    auto camera = scene->GetFirstCameraNode();
    Expect(camera != nullptr, "Billiard camera exists");
    if (camera) {
        const auto eye = *camera->GetCalculatedTransform();
        Expect(eye[3][0] > 1.5f && std::abs(eye[3][1]) < 0.1f &&
                   eye[3][2] >= 1.4f && eye[3][2] <= 1.6f,
               "camera is at player eye height behind cue ball");
        const auto target = camera->GetTarget();
        Expect(std::abs(target[0]) < 0.3f && std::abs(target[1]) < 0.1f &&
                   target[2] > 0.8f && target[2] < 0.9f,
               "camera looks at the playing surface");
        const Vector3f position{eye[3][0], eye[3][1], eye[3][2]};
        auto forward = target - position;
        Normalize(forward);
        Vector3f right, up;
        CrossProduct(right, forward, Vector3f{0.0f, 0.0f, 1.0f});
        Normalize(right);
        CrossProduct(up, right, forward);
        auto object = std::dynamic_pointer_cast<SceneObjectPerspectiveCamera>(
            scene->GetCamera(camera->GetSceneObjectRef()));
        for (float x : {-1.22f, 1.22f}) {
            for (float y : {-0.61f, 0.61f}) {
                const auto offset = Vector3f{x, y, 0.83f} - position;
                const float depth = DotProduct(offset, forward);
                const float halfHeight =
                    depth * std::tan(object->GetFov() / 2.0f);
                Expect(depth > object->GetNearClipDistance() &&
                           std::abs(DotProduct(offset, right)) <
                               halfHeight * 4.0f / 3.0f &&
                           std::abs(DotProduct(offset, up)) < halfHeight,
                       "table corners fit the authored camera at 4:3 aspect");
            }
        }
    }

    TestScenes scenes;
    SteppingBullet physics(scenes);
    Expect(physics.Initialize() == 0, "Billiard Bullet initializes");
    for (const auto& [key, weakNode] : scene->GeometryNodes) {
        auto node = weakNode.lock();
        if (!node) continue;
        auto geometry = scene->GetGeometry(node->GetSceneObjectRef());
        if (geometry->CollisionType() !=
            SceneObjectCollisionType::kSceneObjectCollisionTypeSphere) {
            physics.CreateRigidBody(node, *geometry);
        }
    }
    auto cueGeometry = scene->GetGeometry(cue->GetSceneObjectRef());
    const float restHeight = 0.83f + cueGeometry->CollisionParameters()[0];
    auto simulate = [&](const Vector3f& start, const Vector3f& velocity) {
        physics.DeleteRigidBody(*cue);
        const auto authored = *cue->GetCalculatedTransform();
        cue->MoveBy(start[0] - authored[3][0], start[1] - authored[3][1],
                    start[2] - authored[3][2]);
        physics.CreateRigidBody(cue, *cueGeometry);
        physics.SetVelocity(*cue, velocity);
        for (int i = 0; i < 90; ++i) {
            physics.Step();
            if (cue->GetRenderTransform()[3][2] < 0.6f) break;
        }
        const auto transform = cue->GetRenderTransform();
        return Vector3f{transform[3][0], transform[3][1], transform[3][2]};
    };
    for (const auto& point :
         std::array<Vector3f, 5>{Vector3f{0.0f, 0.0f, restHeight},
                                 Vector3f{0.5f, 0.49f, restHeight},
                                 Vector3f{-0.5f, -0.49f, restHeight},
                                 Vector3f{1.04f, 0.0f, restHeight},
                                 Vector3f{-1.04f, 0.0f, restHeight}}) {
        const auto result = simulate(point, Vector3f{0.0f, 0.0f, 0.0f});
        Expect(std::abs(result[2] - restHeight) < 0.003f,
               "ordinary cloth and support seams retain resting ball");
    }
    for (float side : {-1.0f, 1.0f}) {
        for (float x : {-0.5f, 0.5f}) {
            const auto result = simulate(Vector3f{x, side * 0.35f, restHeight},
                                         Vector3f{0.0f, side * 0.6f, 0.0f});
            Expect(std::abs(result[1]) < 0.51f && result[2] > 0.82f,
                   "each long-edge cushion blocks a rolling ball");
        }
        const auto result = simulate(Vector3f{side * 0.95f, 0.0f, restHeight},
                                     Vector3f{side * 0.6f, 0.0f, 0.0f});
        Expect(std::abs(result[0]) < 1.11f && result[2] > 0.82f,
               "each short-edge cushion blocks a rolling ball");
        const auto middle = simulate(Vector3f{0.0f, side * 0.4f, restHeight},
                                     Vector3f{0.0f, side * 0.3f, 0.0f});
        Expect(middle[2] < 0.6f && std::abs(middle[1]) < 0.61f,
               "each middle pocket allows a rolling ball to fall");
        for (float end : {-1.0f, 1.0f}) {
            const auto corner =
                simulate(Vector3f{end * 0.94f, side * 0.38f, restHeight},
                         Vector3f{end * 0.2f, side * 0.2f, 0.0f});
            Expect(corner[2] < 0.6f && std::abs(corner[0]) < 1.20f &&
                       std::abs(corner[1]) < 0.60f,
                   "each corner pocket allows a rolling ball to fall");
        }
    }
    physics.Finalize();
    scene.reset();
    assets.Finalize();
}

std::shared_ptr<SceneObjectMesh> MakeTriangleMesh(
    const std::vector<float>& positions, const std::vector<uint32_t>& indices) {
    auto mesh = std::make_shared<SceneObjectMesh>();
    auto* vertexData = new uint8_t[positions.size() * sizeof(float)];
    memcpy(vertexData, positions.data(), positions.size() * sizeof(float));
    mesh->AddVertexArray(SceneObjectVertexArray(
        "position", 0, VertexDataType::kVertexDataTypeFloat3, vertexData,
        positions.size()));
    auto* indexData = new uint8_t[indices.size() * sizeof(uint32_t)];
    memcpy(indexData, indices.data(), indices.size() * sizeof(uint32_t));
    mesh->AddIndexArray(SceneObjectIndexArray(
        0, 0, IndexDataType::kIndexDataTypeInt32, indexData, indices.size()));
    mesh->SetPrimitiveType(PrimitiveType::kPrimitiveTypeTriList);
    return mesh;
}

// Drops a ball on a static ground and returns its highest point after the
// first bounce, or the final height if it never rebounds.
float ReboundApex(float restitution, bool tilted, float* lateralDrift) {
    TestScenes scenes;
    SteppingBullet physics(scenes);
    physics.Initialize();

    const std::vector<float> ground =
        tilted ? std::vector<float>{-5, -5, -0.9f, 5, -5, -0.9f, 5, 5, 0.9f,
                                    -5, 5, 0.9f}
               : std::vector<float>{-5, -5, 0, 5, -5, 0, 5, 5, 0, -5, 5, 0};
    SceneObjectGeometry groundGeometry;
    groundGeometry.SetCollisionType(
        SceneObjectCollisionType::kSceneObjectCollisionTypeBvhMesh);
    groundGeometry.AddMesh(MakeTriangleMesh(ground, {0, 1, 2, 0, 2, 3}));
    SceneObjectPhysicsProperties groundMaterial;
    groundMaterial.restitution = 1.0f;
    groundGeometry.SetPhysicsProperties(groundMaterial);
    auto groundNode = std::make_shared<SceneGeometryNode>("ground");
    physics.CreateRigidBody(groundNode, groundGeometry);

    SceneObjectGeometry ballGeometry;
    ballGeometry.SetCollisionType(
        SceneObjectCollisionType::kSceneObjectCollisionTypeSphere);
    const float radius[] = {0.5f};
    ballGeometry.SetCollisionParameters(radius, 1);
    SceneObjectPhysicsProperties ballMaterial;
    ballMaterial.restitution = restitution;
    ballMaterial.planeNormalAxis = 1;
    // Out-of-plane velocity must be discarded by the constraint.
    ballMaterial.linearVelocity = Vector3f{0.0f, 2.0f, 0.0f};
    ballGeometry.SetPhysicsProperties(ballMaterial);
    auto ball = std::make_shared<SceneGeometryNode>("ball");
    ball->MoveBy(0.0f, 0.0f, 4.0f);
    physics.CreateRigidBody(ball, ballGeometry);

    float apex = 0.0f;
    float previous = 4.0f;
    bool bounced = false;
    float drift = 0.0f;
    for (int i = 0; i < 240; ++i) {
        physics.Step();
        const auto transform = ball->GetRenderTransform();
        drift = std::max(drift, std::abs(transform[3][1]));
        if (transform[3][2] > previous && !bounced) bounced = true;
        if (bounced) apex = std::max(apex, transform[3][2]);
        previous = transform[3][2];
    }
    if (lateralDrift) *lateralDrift = drift;
    physics.Finalize();
    return apex;
}

void TestPhysicsMaterialsAndMeshes() {
    float drift = 0.0f;
    const float dull = ReboundApex(0.2f, false, &drift);
    const float lively = ReboundApex(0.9f, false, nullptr);
    Expect(dull > 0.0f && lively > dull + 1.0f,
           "restitution from the physics material scales the rebound");
    Expect(drift < 0.001f,
           "planar constraint discards out-of-plane initial velocity");
    Expect(lively > 0.5f, "triangle mesh ground stops a falling ball");

    float tiltedDrift = 0.0f;
    ReboundApex(0.9f, true, &tiltedDrift);
    Expect(tiltedDrift < 0.001f,
           "planar constraint keeps a ball in plane on a slanted surface");
}

void TestSplashBouncingBall() {
    AssetLoader assets;
    Expect(assets.Initialize() == 0, "Splash assets initialize");
    OgexParser parser;
    auto scene = parser.Parse(
        assets.SyncOpenAndReadTextFileToString("Scene/splash.ogex"));
    Expect(scene != nullptr, "actual Splash scene parses");
    if (!scene) {
        assets.Finalize();
        return;
    }
    auto ball = scene->LUT_Name_GeometryNode.at("Ball").lock();
    auto text = scene->LUT_Name_GeometryNode.at("Text").lock();
    Expect(ball && text, "Splash scene has a ball and text");
    if (!ball || !text) {
        assets.Finalize();
        return;
    }
    Expect(scene->AnimatableNodes.empty(),
           "text stays still so the ball can land on it");

    TestScenes scenes;
    SteppingBullet physics(scenes);
    Expect(physics.Initialize() == 0, "Splash Bullet initializes");
    for (const auto& [key, weakNode] : scene->GeometryNodes) {
        auto node = weakNode.lock();
        if (!node) continue;
        auto geometry = scene->GetGeometry(node->GetSceneObjectRef());
        physics.CreateRigidBody(node, *geometry);
    }
    const float authoredY = (*ball->GetCalculatedTransform())[3][1];

    constexpr float textTop = 0.106f;
    float lowest = 100.0f;
    float drift = 0.0f;
    float previous = ball->GetRenderTransform()[3][2];
    float velocity = 0.0f;
    int bounces = 0;
    int stepsToEnd = -1;
    for (int i = 0; i < 60 * 25 && stepsToEnd < 0; ++i) {
        physics.Step();
        const auto transform = ball->GetRenderTransform();
        const float z = transform[3][2];
        const float next = z - previous;
        if (velocity < -1e-4f && next > 1e-4f) ++bounces;
        if (std::abs(next) > 1e-4f) velocity = next;
        previous = z;
        if (transform[3][0] > -5.4f) lowest = std::min(lowest, z);
        drift = std::max(drift, std::abs(transform[3][1] - authoredY));
        if (transform[3][0] > 5.5f) stepsToEnd = i;
    }
    Expect(stepsToEnd >= 0, "ball reaches the right end of the text");
    Expect(lowest > textTop + 0.15f,
           "ball never drops into the gaps between words");
    Expect(drift < 0.001f, "ball stays in its vertical plane");
    Expect(bounces >= 8, "ball bounces repeatedly along the text");
    physics.Finalize();
    scene.reset();
    assets.Finalize();
}

template <class Physics>
void TestSceneReplacementWithoutApplication() {
    TestScenes scenes;
    Physics physics(scenes);
    Expect(physics.Initialize() == 0, "physics initializes with injected scenes");
    physics.Tick();

    auto node = std::make_shared<SceneGeometryNode>("old");
    auto geometry = std::make_shared<SceneObjectGeometry>();
    geometry->SetCollisionType(SceneObjectCollisionType::kSceneObjectCollisionTypeSphere);
    const float radius[] = {1.0f};
    geometry->SetCollisionParameters(radius, 1);
    node->AddSceneObjectRef("sphere");
    scenes.scene = std::make_shared<Scene>("first");
    scenes.scene->SceneGraph->AppendChild(std::shared_ptr<TreeNode>(node));
    scenes.scene->GeometryNodes.emplace("old", node);
    scenes.scene->Geometries.emplace("sphere", geometry);
    scenes.revision = 1;
    physics.Tick();
    node->MoveBy(3.0f, 0.0f, 0.0f);
    Expect(!Near(node->GetRenderTransform()[3][0], 3.0f),
           "injected scene revision creates a body and publishes transform");
    Expect(physics.UpdateRigidBodyTransform(*node),
           "scene-bound body is addressable without application");

    auto replacementNode = std::make_shared<SceneGeometryNode>("new");
    replacementNode->AddSceneObjectRef("sphere");
    auto replacement = std::make_shared<Scene>("replacement");
    replacement->SceneGraph->AppendChild(std::shared_ptr<TreeNode>(replacementNode));
    replacement->GeometryNodes.emplace("new", replacementNode);
    replacement->Geometries.emplace("sphere", geometry);
    scenes.scene.reset();
    scenes.revision = 2;
    physics.Tick();
    node->MoveBy(2.0f, 0.0f, 0.0f);
    Expect(Near(node->GetRenderTransform()[3][0], 5.0f) &&
               !physics.UpdateRigidBodyTransform(*node),
           "missing replacement removes old body and clears its simulation transform");
    scenes.scene = replacement;
    physics.Tick();
    Expect(physics.UpdateRigidBodyTransform(*replacementNode),
           "same revision is retried and creates replacement body");
    physics.Finalize();
}
}  // namespace

int main() {
    TestBindingsAndTransforms<MyPhysicsManager>();
    TestBindingsAndTransforms<SteppingBullet>();
    TestBulletSimulationPublication();
    TestBilliardScene();
    TestPhysicsMaterialsAndMeshes();
    TestSplashBouncingBall();
    TestSceneReplacementWithoutApplication<MyPhysicsManager>();
    TestSceneReplacementWithoutApplication<SteppingBullet>();
    return failures == 0 ? 0 : 1;
}
