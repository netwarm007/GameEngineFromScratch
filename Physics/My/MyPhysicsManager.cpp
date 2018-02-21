#include <iostream>
#include "MyPhysicsManager.hpp"
#include "Box.hpp"
#include "Plane.hpp"
#include "Sphere.hpp"
#include "RigidBody.hpp"
#include "GraphicsManager.hpp"

using namespace My;
using namespace std;

int MyPhysicsManager::Initialize()
{
    cout << "[MyPhysicsManager] Initialize" << endl;
    return 0;
}

void MyPhysicsManager::Finalize()
{
    cout << "[MyPhysicsManager] Finalize" << endl;
    // Clean up
    ClearRigidBodies();
}

void MyPhysicsManager::Tick()
{
    if (g_pSceneManager->IsSceneChanged())
    {
        ClearRigidBodies();
        CreateRigidBodies();
        g_pSceneManager->NotifySceneIsPhysicalSimulationQueued();
    }
}

void MyPhysicsManager::CreateRigidBody(SceneGeometryNode& node, const SceneObjectGeometry& geometry)
{
    const float* param = geometry.CollisionParameters();
    RigidBody* rigidBody = nullptr;

    switch(geometry.CollisionType())
    {
        case SceneObjectCollisionType::kSceneObjectCollisionTypeSphere:
            {
                auto collision_box = make_shared<Sphere>(param[0]);

                const auto trans = node.GetCalculatedTransform();
                auto motionState = 
                    make_shared<MotionState>(
                                *trans 
                            );
                rigidBody = new RigidBody(collision_box, motionState);
            }
            break;
        case SceneObjectCollisionType::kSceneObjectCollisionTypeBox:
            {
                auto collision_box = make_shared<Box>(Vector3f(param[0], param[1], param[2]));

                const auto trans = node.GetCalculatedTransform();
                auto motionState = 
                    make_shared<MotionState>(
                                *trans 
                            );
                rigidBody = new RigidBody(collision_box, motionState);
            }
            break;
        case SceneObjectCollisionType::kSceneObjectCollisionTypePlane:
            {
                auto collision_box = make_shared<Plane>(Vector3f(param[0], param[1], param[2]), param[3]);

                const auto trans = node.GetCalculatedTransform();
                auto motionState = 
                    make_shared<MotionState>(
                                *trans 
                            );
                rigidBody = new RigidBody(collision_box, motionState);
            }
            break;
        default:
            {
                // create AABB box according to Bounding Box 
                auto collision_box = make_shared<Box>(geometry.GetBoundingBox());

                const auto trans = node.GetCalculatedTransform();
                auto motionState = 
                    make_shared<MotionState>(
                                *trans 
                            );
                rigidBody = new RigidBody(collision_box, motionState);
            }
    }

    node.LinkRigidBody(rigidBody);
}

void MyPhysicsManager::UpdateRigidBodyTransform(SceneGeometryNode& node)
{
    const auto trans = node.GetCalculatedTransform();
    auto rigidBody = node.RigidBody();
    auto motionState = reinterpret_cast<RigidBody*>(rigidBody)->GetMotionState();
    motionState->SetTransition(*trans);
}

void MyPhysicsManager::DeleteRigidBody(SceneGeometryNode& node)
{
    RigidBody* rigidBody = reinterpret_cast<RigidBody*>(node.UnlinkRigidBody());
    if(rigidBody) {
        delete rigidBody;
    }
}

int MyPhysicsManager::CreateRigidBodies()
{
    auto& scene = g_pSceneManager->GetSceneForPhysicalSimulation();

    // Geometries
    for (auto _it : scene.GeometryNodes)
    {
        auto pGeometryNode = _it.second;
        auto pGeometry = scene.GetGeometry(pGeometryNode->GetSceneObjectRef());
        assert(pGeometry);

        CreateRigidBody(*pGeometryNode, *pGeometry);
    }

    return 0;
}

void MyPhysicsManager::ClearRigidBodies()
{
    auto& scene = g_pSceneManager->GetSceneForPhysicalSimulation();

    // Geometries
    for (auto _it : scene.GeometryNodes)
    {
        auto pGeometryNode = _it.second;
        DeleteRigidBody(*pGeometryNode);
    }
}

Matrix4X4f MyPhysicsManager::GetRigidBodyTransform(void* rigidBody)
{
    Matrix4X4f trans;

    RigidBody* _rigidBody = reinterpret_cast<RigidBody*>(rigidBody);
    auto motionState = _rigidBody->GetMotionState();
    trans = motionState->GetTransition();

    return trans;
}

void MyPhysicsManager::ApplyCentralForce(void* rigidBody, Vector3f force)
{

}

#ifdef DEBUG
    void MyPhysicsManager::DrawDebugInfo()
    {
        auto& scene = g_pSceneManager->GetSceneForPhysicalSimulation();

        // Geometries
        for (auto _it : scene.GeometryNodes)
        {
            auto pGeometryNode = _it.second;
            if (void* rigidBody = pGeometryNode->RigidBody()) {
                RigidBody* _rigidBody = reinterpret_cast<RigidBody*>(rigidBody);
                Matrix4X4f simulated_result = GetRigidBodyTransform(_rigidBody);
                auto pGeometry = _rigidBody->GetCollisionShape();
                DrawAabb(*pGeometry, simulated_result);
            }
        }
    }

    void MyPhysicsManager::DrawAabb(const Geometry& geometry, const Matrix4X4f& trans)
    {
        Vector3f bbMin, bbMax;
        Vector3f color(0.5f, 0.5f, 0.5f);

        geometry.GetAabb(trans, bbMin, bbMax);
        g_pGraphicsManager->DrawBox(bbMin, bbMax, color);
    }
#endif