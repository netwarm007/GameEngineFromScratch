#include <iostream>
#include "MyPhysicsManager.hpp"

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
}

void MyPhysicsManager::Tick()
{

}

void MyPhysicsManager::CreateRigidBody(SceneGeometryNode& node, const SceneObjectGeometry& geometry)
{

}

void MyPhysicsManager::DeleteRigidBody(SceneGeometryNode& node)
{

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
