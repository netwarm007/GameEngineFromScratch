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
}

void MyPhysicsManager::IterateConvexHull() {
    auto pSceneManager =
        dynamic_cast<BaseApplication*>(m_pApp)->GetSceneManager();
    auto& scene = pSceneManager->GetSceneForPhysicalSimulation();

    // Geometries
    for (const auto& _it : scene->GeometryNodes) {
        auto pGeometryNode = _it.second.lock();
        if (pGeometryNode) {
            void* rigidBody = pGeometryNode->RigidBody();
            if (rigidBody) {
                auto* _rigidBody = reinterpret_cast<RigidBody*>(rigidBody);
                auto pGeometry = _rigidBody->GetCollisionShape();
                if (pGeometry->GetGeometryType() == GeometryType::kPolyhydron) {
                    if (dynamic_pointer_cast<ConvexHull<float>>(pGeometry)
                            ->Iterate()) {
                        // The geometry convex hull is not fully iterated,
                        // so we break the loop to postpending the process to
                        // next loop, to avoid too much process in single tick.
                        break;
                    }

                    // The geometry convex hull is fully iterated,
                    // so we move to next geometry
                }
            }
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
}

void MyPhysicsManager::CreateRigidBody(SceneGeometryNode& node,
                                       const SceneObjectGeometry& geometry) {
    const float* param = geometry.CollisionParameters();
    RigidBody* rigidBody = nullptr;

    switch (geometry.CollisionType()) {
        case SceneObjectCollisionType::kSceneObjectCollisionTypeSphere: {
            auto collision_box = make_shared<Sphere<float>>(param[0]);

            const auto trans = node.GetCalculatedTransform();
            auto motionState = make_shared<MotionState>(*trans);
            rigidBody = new RigidBody(collision_box, motionState);
        } break;
        case SceneObjectCollisionType::kSceneObjectCollisionTypeBox: {
            auto collision_box =
                make_shared<Box<float>>(Vector3f({param[0], param[1], param[2]}));

            const auto trans = node.GetCalculatedTransform();
            auto motionState = make_shared<MotionState>(*trans);
            rigidBody = new RigidBody(collision_box, motionState);
        } break;
        case SceneObjectCollisionType::kSceneObjectCollisionTypePlane: {
            auto collision_box = make_shared<Plane>(
                Vector3f({param[0], param[1], param[2]}), param[3]);

            const auto trans = node.GetCalculatedTransform();
            auto motionState = make_shared<MotionState>(*trans);
            rigidBody = new RigidBody(collision_box, motionState);
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

    node.LinkRigidBody(rigidBody);
}

void MyPhysicsManager::UpdateRigidBodyTransform(SceneGeometryNode& node) {
    const auto trans = node.GetCalculatedTransform();
    auto rigidBody = node.RigidBody();
    auto motionState =
        reinterpret_cast<RigidBody*>(rigidBody)->GetMotionState();
    motionState->SetTransition(*trans);
}

void MyPhysicsManager::DeleteRigidBody(SceneGeometryNode& node) {
    auto* rigidBody = reinterpret_cast<RigidBody*>(node.UnlinkRigidBody());
    delete rigidBody;
}

int MyPhysicsManager::CreateRigidBodies() {
    auto pSceneManager =
        dynamic_cast<BaseApplication*>(m_pApp)->GetSceneManager();
    auto& scene = pSceneManager->GetSceneForPhysicalSimulation();

    // Geometries
    for (const auto& _it : scene->GeometryNodes) {
        auto pGeometryNode = _it.second.lock();
        if (pGeometryNode) {
            auto pGeometry =
                scene->GetGeometry(pGeometryNode->GetSceneObjectRef());
            assert(pGeometry);

            CreateRigidBody(*pGeometryNode, *pGeometry);
        }
    }

    return 0;
}

void MyPhysicsManager::ClearRigidBodies() {
    auto pSceneManager =
        dynamic_cast<BaseApplication*>(m_pApp)->GetSceneManager();
    auto& scene = pSceneManager->GetSceneForPhysicalSimulation();

    // Geometries
    for (const auto& _it : scene->GeometryNodes) {
        auto pGeometryNode = _it.second.lock();
        if (pGeometryNode) {
            DeleteRigidBody(*pGeometryNode);
        }
    }
}

Matrix4X4f MyPhysicsManager::GetRigidBodyTransform(void* rigidBody) {
    auto* _rigidBody = reinterpret_cast<RigidBody*>(rigidBody);
    auto motionState = _rigidBody->GetMotionState();
    return motionState->GetTransition();
}

void MyPhysicsManager::ApplyCentralForce(void* rigidBody, Vector3f force) {}
