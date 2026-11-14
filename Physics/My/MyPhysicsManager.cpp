#include "MyPhysicsManager.hpp"

#include <iostream>

#include "BaseApplication.hpp"
#include "Box.hpp"
#include "Plane.hpp"
#include "RigidBody.hpp"
#include "Sphere.hpp"

using namespace My;
using namespace std;

int MyPhysicsManager::Initialize() {
    cout << "[MyPhysicsManager] Initialize" << endl;
    return 0;
}

void MyPhysicsManager::Finalize() {
    cout << "[MyPhysicsManager] Finalize" << endl;
    // Clean up
    ClearRigidBodies();
    m_nSceneRevision = 0;
}

void MyPhysicsManager::IterateConvexHull() {
    for (const auto& [node, binding] : m_RigidBodies) {
        auto geometry = binding.body->GetCollisionShape();
        if (geometry->GetGeometryType() == GeometryType::kPolyhydron &&
            dynamic_pointer_cast<ConvexHull<float_precision>>(geometry)->Iterate()) {
            // Continue unfinished hulls on the next tick.
            break;
        }
    }
}

void MyPhysicsManager::Tick() {
    auto pSceneManager =
        dynamic_cast<BaseApplication*>(m_pApp)->GetSceneManager();
    auto rev = pSceneManager->GetSceneRevision();
    if (m_nSceneRevision != rev) {
        ClearRigidBodies();
        CreateRigidBodies();
        m_nSceneRevision = rev;
    }
    for (const auto& [node, binding] : m_RigidBodies) {
        node->SetSimulationTransform(binding.body->GetMotionState()->GetTransition());
    }
}

void MyPhysicsManager::CreateRigidBody(
    const std::shared_ptr<SceneGeometryNode>& sceneNode,
    const SceneObjectGeometry& geometry) {
    assert(sceneNode);
    auto& node = *sceneNode;
    DeleteRigidBody(node);
    const float* param = geometry.CollisionParameters();
    RigidBody<float_precision>* rigidBody = nullptr;

    switch (geometry.CollisionType()) {
        case SceneObjectCollisionType::kSceneObjectCollisionTypeSphere: {
            auto collision_box = make_shared<Sphere<float, void*>>(param[0]);

            const auto trans = node.GetCalculatedTransform();
            auto motionState = make_shared<MotionState>(*trans);
            rigidBody = new RigidBody<float_precision>(collision_box, motionState);
        } break;
        case SceneObjectCollisionType::kSceneObjectCollisionTypeBox: {
            auto collision_box =
                make_shared<Box<float_precision>>(Vector3<float_precision>({param[0], param[1], param[2]}));

            const auto trans = node.GetCalculatedTransform();
            auto motionState = make_shared<MotionState>(*trans);
            rigidBody = new RigidBody<float_precision>(collision_box, motionState);
        } break;
        case SceneObjectCollisionType::kSceneObjectCollisionTypePlane: {
            auto collision_box = make_shared<Plane<float_precision>>(
                Vector3f({param[0], param[1], param[2]}), param[3]);

            const auto trans = node.GetCalculatedTransform();
            auto motionState = make_shared<MotionState>(*trans);
            rigidBody = new RigidBody<float_precision>(collision_box, motionState);
        } break;
        default: {
            /*
            // create collision box using convex hull
            auto bounding_box = geometry.GetBoundingBox();
            auto collision_box =
            make_shared<ConvexHull>(geometry.GetConvexHull());

            const auto trans = node.GetCalculatedTransform();
            auto motionState =
                make_shared<MotionState>(
                            *trans,
                            bounding_box.centroid
                        );
            rigidBody = new RigidBody(collision_box, motionState);
            */
        }
    }

    if (rigidBody) {
        m_RigidBodies.emplace(
            &node, BodyBinding{sceneNode,
                             std::unique_ptr<RigidBody<float_precision>>(rigidBody)});
        node.SetSimulationTransform(rigidBody->GetMotionState()->GetTransition());
    }
}

bool MyPhysicsManager::UpdateRigidBodyTransform(SceneGeometryNode& node) {
    auto it = m_RigidBodies.find(&node);
    if (it == m_RigidBodies.end()) {
        return false;
    }
    const auto trans = node.GetCalculatedTransform();
    auto motionState = it->second.body->GetMotionState();
    motionState->SetTransition(*trans);
    node.SetSimulationTransform(*trans);
    return true;
}

void MyPhysicsManager::DeleteRigidBody(SceneGeometryNode& node) {
    node.ClearSimulationTransform();
    m_RigidBodies.erase(&node);
}

int MyPhysicsManager::CreateRigidBodies() {
    ClearRigidBodies();
    auto pSceneManager =
        dynamic_cast<BaseApplication*>(m_pApp)->GetSceneManager();
    m_pSimulationScene = pSceneManager->GetSceneForPhysicalSimulation();
    const auto& scene = m_pSimulationScene;

    // Geometries
    for (const auto& _it : scene->GeometryNodes) {
        auto pGeometryNode = _it.second.lock();
        if (pGeometryNode) {
            auto pGeometry =
                scene->GetGeometry(pGeometryNode->GetSceneObjectRef());
            assert(pGeometry);

            CreateRigidBody(pGeometryNode, *pGeometry);
        }
    }

    return 0;
}

void MyPhysicsManager::ClearRigidBodies() {
    while (!m_RigidBodies.empty()) {
        DeleteRigidBody(*m_RigidBodies.begin()->first);
    }
    m_pSimulationScene.reset();
}

bool MyPhysicsManager::ApplyCentralForce(SceneGeometryNode& node, Vector3f force) {
    return false;
}
