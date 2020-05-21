#include "MyPhysicsManager.hpp"

#include <iostream>

#include "Box.hpp"
#include "GraphicsManager.hpp"
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
    auto& scene = g_pSceneManager->GetSceneForPhysicalSimulation();

    // Geometries
    for (const auto& _it : scene->GeometryNodes) {
        auto pGeometryNode = _it.second.lock();
        if (pGeometryNode) {
            void* rigidBody = pGeometryNode->RigidBody();
            if (rigidBody) {
                auto* _rigidBody = reinterpret_cast<RigidBody*>(rigidBody);
                auto pGeometry = _rigidBody->GetCollisionShape();
                if (pGeometry->GetGeometryType() == GeometryType::kPolyhydron) {
                    if (dynamic_pointer_cast<ConvexHull>(pGeometry)
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
    auto rev = g_pSceneManager->GetSceneRevision();
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
            auto collision_box = make_shared<Sphere>(param[0]);

            const auto trans = node.GetCalculatedTransform();
            auto motionState = make_shared<MotionState>(*trans);
            rigidBody = new RigidBody(collision_box, motionState);
        } break;
        case SceneObjectCollisionType::kSceneObjectCollisionTypeBox: {
            auto collision_box =
                make_shared<Box>(Vector3f({param[0], param[1], param[2]}));

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
    auto& scene = g_pSceneManager->GetSceneForPhysicalSimulation();

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
    auto& scene = g_pSceneManager->GetSceneForPhysicalSimulation();

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

#ifdef DEBUG
void MyPhysicsManager::DrawDebugInfo() {
    auto& scene = g_pSceneManager->GetSceneForPhysicalSimulation();

    // Geometries
    for (const auto& _it : scene->GeometryNodes) {
        auto pGeometryNode = _it.second.lock();
        if (pGeometryNode) {
            if (void* rigidBody = pGeometryNode->RigidBody()) {
                auto* _rigidBody = reinterpret_cast<RigidBody*>(rigidBody);
                auto motionState = _rigidBody->GetMotionState();
                auto centerOfMass = motionState->GetCenterOfMassOffset();
                auto trans = motionState->GetTransition();
                auto pGeometry = _rigidBody->GetCollisionShape();
                DrawAabb(*pGeometry, trans, centerOfMass);
                DrawShape(*pGeometry, trans, centerOfMass);
            }
        }
    }
}

void MyPhysicsManager::DrawAabb(const Geometry& geometry,
                                const Matrix4X4f& trans,
                                const Vector3f& centerOfMass) {
    Vector3f bbMin, bbMax;
    Vector3f color({0.7f, 0.6f, 0.5f});

    Matrix4X4f _trans;
    BuildIdentityMatrix(_trans);
    _trans.data[3][0] = centerOfMass[0] * trans.data[0][0];  // scale by x-scale
    _trans.data[3][1] = centerOfMass[1] * trans.data[1][1];  // scale by y-scale
    _trans.data[3][2] = centerOfMass[2] * trans.data[2][2];  // scale by z-scale
    MatrixMultiply(_trans, trans, _trans);

    geometry.GetAabb(_trans, bbMin, bbMax);
    g_pGraphicsManager->DrawBox(bbMin, bbMax, color);
}

void MyPhysicsManager::DrawShape(const Geometry& geometry,
                                 const Matrix4X4f& trans,
                                 const Vector3f& centerOfMass) {
    Vector3f color({0.8f, 0.7f, 0.6f});

    Matrix4X4f _trans;
    BuildIdentityMatrix(_trans);
    _trans.data[3][0] = centerOfMass[0] * trans.data[0][0];  // scale by x-scale
    _trans.data[3][1] = centerOfMass[1] * trans.data[1][1];  // scale by y-scale
    _trans.data[3][2] = centerOfMass[2] * trans.data[2][2];  // scale by z-scale
    MatrixMultiply(_trans, trans, _trans);

    if (geometry.GetGeometryType() == GeometryType::kPolyhydron) {
        g_pGraphicsManager->DrawPolyhydron(
            reinterpret_cast<const Polyhedron&>(geometry), trans, color);
    }
}
#endif