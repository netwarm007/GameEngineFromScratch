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

void article_44_logic::OnLeftKey()
{

}

void article_44_logic::OnRightKey()
{

}

void article_44_logic::OnUpKey()
{

}

void article_44_logic::OnDownKey()
{

}
