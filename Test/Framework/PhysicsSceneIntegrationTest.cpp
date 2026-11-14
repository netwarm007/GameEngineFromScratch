#include <cmath>
#include <iostream>
#include <memory>

#include "Bullet/BulletPhysicsManager.hpp"
#include "ISceneManager.hpp"
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
    TestSceneReplacementWithoutApplication<MyPhysicsManager>();
    TestSceneReplacementWithoutApplication<SteppingBullet>();
    return failures == 0 ? 0 : 1;
}
