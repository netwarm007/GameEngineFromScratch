#include <iostream>
#include "MyPhysicsManager.hpp"
#include "Box.hpp"
#include "Plane.hpp"
#include "Sphere.hpp"
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
    Geometry* collision_box = nullptr;

    switch(geometry.CollisionType())
    {
        case SceneObjectCollisionType::kSceneObjectCollisionTypeSphere:
            {
                collision_box = new Sphere(param[0]);
                m_CollisionShapes.push_back(collision_box);
            }
            break;
        case SceneObjectCollisionType::kSceneObjectCollisionTypeBox:
            {
                collision_box = new Box(Vector3f(param[0], param[1], param[2]));
                m_CollisionShapes.push_back(collision_box);
            }
            break;
        case SceneObjectCollisionType::kSceneObjectCollisionTypePlane:
            {
                collision_box = new Plane(Vector3f(param[0], param[1], param[2]), param[3]);
                m_CollisionShapes.push_back(collision_box);
            }
            break;
        default:
            ;
    }

    node.LinkRigidBody(collision_box);
}

void MyPhysicsManager::DeleteRigidBody(SceneGeometryNode& node)
{
    Geometry* collision_box = reinterpret_cast<Geometry*>(node.UnlinkRigidBody());
    if(collision_box) {
        delete collision_box;
    }
}

int MyPhysicsManager::CreateRigidBodies()
{
    auto& scene = g_pSceneManager->GetSceneForRendering();

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
    auto& scene = g_pSceneManager->GetSceneForRendering();

    // Geometries
    for (auto _it : scene.GeometryNodes)
    {
        auto pGeometryNode = _it.second;
        DeleteRigidBody(*pGeometryNode);
    }

    for (auto shape : m_CollisionShapes)
    {
        delete shape;
    }

    m_CollisionShapes.clear();
}

Matrix4X4f MyPhysicsManager::GetRigidBodyTransform(void* RigidBody)
{
    Matrix4X4f trans;

    return trans;
}

void MyPhysicsManager::ApplyCentralForce(void* rigidBody, Vector3f force)
{

}

#ifdef DEBUG
void MyPhysicsManager::DrawDebugInfo()
{
    Vector3f from (-10.0f, 0.0f, 0.0f);
    Vector3f to (10.0f, 0.0f, 0.0f);
    Vector3f color(1.0f, 0.0f, 0.0f);
    g_pGraphicsManager->DrawLine(from, to, color);
}
#endif