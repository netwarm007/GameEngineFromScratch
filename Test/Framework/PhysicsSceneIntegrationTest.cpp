#include <array>
#include <cmath>
#include <iostream>
#include <memory>

#include "AssetLoader.hpp"
#include "Bullet/BulletPhysicsManager.hpp"
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

class SteppingBullet : public BulletPhysicsManager {
   public:
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
    Physics physics;
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
    SteppingBullet physics;
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

    SteppingBullet physics;
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
}  // namespace

int main() {
    TestBindingsAndTransforms<MyPhysicsManager>();
    TestBindingsAndTransforms<SteppingBullet>();
    TestBulletSimulationPublication();
    TestBilliardScene();
    return failures == 0 ? 0 : 1;
}
