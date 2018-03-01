#include <iostream>
#include <random>
#include "article_44_logic.hpp"
#include "GraphicsManager.hpp"
#include "SceneManager.hpp"
#include "IPhysicsManager.hpp"

using namespace My;
using namespace std;

int article_44_logic::Initialize()
{
    int result;

    cout << "[GameLogic] Initialize" << endl;
    cout << "[GameLogic] Start Loading Game Scene" << endl;
    result = g_pSceneManager->LoadScene("Scene/Empty.ogex");

    // generate random point cloud
    default_random_engine generator;
    uniform_real_distribution<float> distribution(-3.0f, 3.0f);
    auto dice = std::bind(distribution, generator);

    for(auto i = 0; i < 100; i++)
    {
        PointPtr point_ptr = make_shared<Point>(dice(), dice(), dice());
        m_QuickHull.AddPoint(std::move(point_ptr));
    }

    m_QuickHull.ComputeHull();

    return result;
}

void article_44_logic::Finalize()
{
    cout << "[GameLogic] Finalize" << endl;
}

void article_44_logic::Tick()
{

}

void article_44_logic::DrawDebugInfo()
{
    auto point_set = m_QuickHull.GetPointSet();
    auto hull = m_QuickHull.GetHull();

    // draw the hull
    g_pGraphicsManager->DrawPolyhydron(hull, Vector3f(0.9f, 0.5f, 0.5f));

    // draw the point cloud
    g_pGraphicsManager->DrawPointSet(point_set, Vector3f(0.5f));
}

void article_44_logic::OnLeftKeyDown()
{
    auto& scene = g_pSceneManager->GetSceneForRendering();
    auto pCameraNode = scene.GetFirstCameraNode();
    if (pCameraNode) {
        auto local_axis = pCameraNode->GetLocalAxis(); 
        Vector3f camera_x_axis;
        memcpy(camera_x_axis.data, local_axis[0], sizeof(camera_x_axis));

        // move camera along its local axis x direction
        pCameraNode->MoveBy(camera_x_axis);
    }
}

void article_44_logic::OnRightKeyDown()
{
    auto& scene = g_pSceneManager->GetSceneForRendering();
    auto pCameraNode = scene.GetFirstCameraNode();
    if (pCameraNode) {
        auto local_axis = pCameraNode->GetLocalAxis(); 
        Vector3f camera_x_axis;
        memcpy(camera_x_axis.data, local_axis[0], sizeof(camera_x_axis));

        // move along camera local axis -x direction
        pCameraNode->MoveBy(camera_x_axis * -1.0f);
    }
}

void article_44_logic::OnUpKeyDown()
{
    auto& scene = g_pSceneManager->GetSceneForRendering();
    auto pCameraNode = scene.GetFirstCameraNode();
    if (pCameraNode) {
        auto local_axis = pCameraNode->GetLocalAxis(); 
        Vector3f camera_y_axis;
        memcpy(camera_y_axis.data, local_axis[1], sizeof(camera_y_axis));

        // move camera along its local axis y direction
        pCameraNode->MoveBy(camera_y_axis);
    }
}

void article_44_logic::OnDownKeyDown()
{
    auto& scene = g_pSceneManager->GetSceneForRendering();
    auto pCameraNode = scene.GetFirstCameraNode();
    if (pCameraNode) {
        auto local_axis = pCameraNode->GetLocalAxis(); 
        Vector3f camera_y_axis;
        memcpy(camera_y_axis.data, local_axis[1], sizeof(camera_y_axis));

        // move camera along its local axis -y direction
        pCameraNode->MoveBy(camera_y_axis * -1.0f);
    }
}
