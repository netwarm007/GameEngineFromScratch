#include <cmath>
#include <iostream>
#include <memory>

#include "Bullet/BulletPhysicsManager.hpp"
#include "My/MyPhysicsManager.hpp"

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
}  // namespace

int main() {
    TestBindingsAndTransforms<MyPhysicsManager>();
    TestBindingsAndTransforms<SteppingBullet>();
    TestBulletSimulationPublication();
    return failures == 0 ? 0 : 1;
}
