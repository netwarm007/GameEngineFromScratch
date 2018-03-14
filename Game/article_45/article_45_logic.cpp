#include <iostream>
#include <random>
#include "article_45_logic.hpp"
#include "IApplication.hpp"
#include "GraphicsManager.hpp"
#include "SceneManager.hpp"
#include "IPhysicsManager.hpp"
#include "Gjk.hpp"

using namespace My;
using namespace std;
using namespace std::placeholders;

int article_45_logic::Initialize()
{
    int result;

    cout << "[GameLogic] Initialize" << endl;
    cout << "[GameLogic] Start Loading Game Scene" << endl;
    result = g_pSceneManager->LoadScene("Scene/Empty.ogex");

    int point_count = 30;
    if (g_pApp->GetCommandLineArgumentsCount() > 1)
        point_count = atoi(g_pApp->GetCommandLineArgument(1));

    // generate random point cloud
    {
        default_random_engine generator;
        uniform_real_distribution<float> distribution(-5.0f, 2.0f);
        auto dice = std::bind(distribution, generator);

        generator.seed(1);
        for(auto i = 0; i < point_count; i++)
        {
            PointPtr point_ptr = make_shared<Point>(dice(), dice(), dice());
            m_QuickHullA.AddPoint(std::move(point_ptr));
        }
    }

    {
        default_random_engine generator;
        uniform_real_distribution<float> distribution(0.5f, 5.0f);
        auto dice = std::bind(distribution, generator);

        generator.seed(300);
        for(auto i = 0; i < point_count; i++)
        {
            PointPtr point_ptr = make_shared<Point>(dice(), dice(), dice());
            m_QuickHullB.AddPoint(std::move(point_ptr));
        }
    }

    return result;
}

void article_45_logic::Finalize()
{
    cout << "[GameLogic] Finalize" << endl;
}

void article_45_logic::Tick()
{
    auto A = m_QuickHullA.GetHull();
    auto B = m_QuickHullB.GetHull();

    if (A.Faces.size() > 3 && B.Faces.size() > 3)
    {
        SupportFunction support_function_A = std::bind(ConvexPolyhedronSupportFunction, A, _1);
        SupportFunction support_function_B = std::bind(ConvexPolyhedronSupportFunction, B, _1);
        PointList simplex;
        Vector3f direction (1.0f, 0.0f, 0.0f);
        int intersected;
        while ((intersected = GjkIntersection(support_function_A, support_function_B, direction, simplex)) == -1)
            ;
        m_bCollided = (intersected == 1)?true:false;
    }
}

#ifdef DEBUG
void article_45_logic::DrawDebugInfo()
{
    auto point_set = m_QuickHullA.GetPointSet();
    auto hull = m_QuickHullA.GetHull();
    Vector3f color_A (0.9f, 0.5f, 0.5f);
    Vector3f color_B (0.5f, 0.5f, 0.9f);
    Vector3f color_collided (0.9f, 0.8f, 0.0f);

    // draw the hull A
    g_pGraphicsManager->DrawPolyhydron(hull, (m_bCollided?color_collided : color_A));

    // draw the point cloud A
    g_pGraphicsManager->DrawPointSet(point_set, Vector3f(0.5f, 0.1f, 0.1f));

    point_set = m_QuickHullB.GetPointSet();
    hull = m_QuickHullB.GetHull();

    // draw the hull B
    g_pGraphicsManager->DrawPolyhydron(hull, (m_bCollided?color_collided : color_B));

    // draw the point cloud B
    g_pGraphicsManager->DrawPointSet(point_set, Vector3f(0.1f, 0.1f, 0.5f));
}
#endif

void article_45_logic::OnLeftKeyDown()
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

void article_45_logic::OnRightKeyDown()
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

void article_45_logic::OnUpKeyDown()
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

void article_45_logic::OnDownKeyDown()
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

void article_45_logic::OnButton1Down()
{
    static bool first_time = true;

    if (first_time)
    {
        m_QuickHullA.Init();
        m_QuickHullB.Init();
        first_time = false;
    }

    m_QuickHullA.Iterate();
    m_QuickHullB.Iterate();
}

void article_45_logic::OnAnalogStick(int id, float deltaX, float deltaY)
{
    auto& scene = g_pSceneManager->GetSceneForRendering();
    auto pCameraNode = scene.GetFirstCameraNode();
    if (pCameraNode) {
        auto screen_width = g_pApp->GetConfiguration().screenWidth;
        auto screen_height = g_pApp->GetConfiguration().screenHeight;
        // move camera along its local axis -y direction
        pCameraNode->RotateBy(deltaX / screen_width * PI, deltaY / screen_height * PI, 0.0f);
    }
}
