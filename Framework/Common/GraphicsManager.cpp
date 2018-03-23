#include <iostream>
#include "GraphicsManager.hpp"
#include "SceneManager.hpp"
#include "cbuffer.h"
#include "IApplication.hpp"
#include "SceneManager.hpp"

using namespace My;
using namespace std;

int GraphicsManager::Initialize()
{
    int result = 0;
	InitConstants();
    return result;
}

void GraphicsManager::Finalize()
{
#ifdef DEBUG
    ClearDebugBuffers();
#endif
    ClearBuffers();
    ClearShaders();
}

void GraphicsManager::Tick()
{
    if (g_pSceneManager->IsSceneChanged())
    {
        cout << "[GraphicsManager] Detected Scene Change, reinitialize buffers ..." << endl;
        ClearBuffers();
        ClearShaders();
        const Scene& scene = g_pSceneManager->GetSceneForRendering();
        InitializeShaders();
        InitializeBuffers(scene);
        g_pSceneManager->NotifySceneIsRenderingQueued();
    }

    UpdateConstants();

    Clear();
    Draw();
}

void GraphicsManager::UpdateConstants()
{
    // Generate the view matrix based on the camera's position.
    CalculateCameraMatrix();
    CalculateLights();
}

void GraphicsManager::Clear()
{

}

void GraphicsManager::Draw()
{
    UpdateConstants();

    RenderBuffers();
#ifdef DEBUG
    RenderDebugBuffers();
#endif
}

void GraphicsManager::InitConstants()
{
    // Initialize the world/model matrix to the identity matrix.
    BuildIdentityMatrix(m_DrawFrameContext.m_worldMatrix);
}

bool GraphicsManager::InitializeShaders()
{
    cout << "[GraphicsManager] GraphicsManager::InitializeShader()" << endl;
    return true;
}

void GraphicsManager::ClearShaders()
{
    cout << "[GraphicsManager] GraphicsManager::ClearShaders()" << endl;
}

void GraphicsManager::CalculateCameraMatrix()
{
    auto& scene = g_pSceneManager->GetSceneForRendering();
    auto pCameraNode = scene.GetFirstCameraNode();
    if (pCameraNode) {
        auto transform = *pCameraNode->GetCalculatedTransform();
        InverseMatrix4X4f(transform);
        m_DrawFrameContext.m_viewMatrix = transform;
    }
    else {
        // use default build-in camera
        Vector3f position = { 0.0f, -5.0f, 0.0f }, lookAt = { 0.0f, 0.0f, 0.0f }, up = { 0.0f, 0.0f, 1.0f };
        BuildViewMatrix(m_DrawFrameContext.m_viewMatrix, position, lookAt, up);
    }

    float fieldOfView = PI / 2.0f;
    float nearClipDistance = 1.0f;
    float farClipDistance = 100.0f;

    if (pCameraNode) {
        auto pCamera = scene.GetCamera(pCameraNode->GetSceneObjectRef());
        // Set the field of view and screen aspect ratio.
        fieldOfView = dynamic_pointer_cast<SceneObjectPerspectiveCamera>(pCamera)->GetFov();
        nearClipDistance = pCamera->GetNearClipDistance();
        farClipDistance = pCamera->GetFarClipDistance();
    }

    const GfxConfiguration& conf = g_pApp->GetConfiguration();

    float screenAspect = (float)conf.screenWidth / (float)conf.screenHeight;

    // Build the perspective projection matrix.
    BuildPerspectiveFovRHMatrix(m_DrawFrameContext.m_projectionMatrix, fieldOfView, screenAspect, nearClipDistance, farClipDistance);
}

void GraphicsManager::CalculateLights()
{
    m_DrawFrameContext.m_ambientColor = {0.01f, 0.01f, 0.01f};

    auto& scene = g_pSceneManager->GetSceneForRendering();
    auto pLightNode = scene.GetFirstLightNode();
    if (pLightNode) {
        m_DrawFrameContext.m_lightPosition = { 0.0f, 0.0f, 0.0f };
        TransformCoord(m_DrawFrameContext.m_lightPosition, *pLightNode->GetCalculatedTransform());

        auto pLight = scene.GetLight(pLightNode->GetSceneObjectRef());
        if (pLight) {
            m_DrawFrameContext.m_lightColor = pLight->GetColor().Value;
        }
    }
    else {
        // use default build-in light 
        m_DrawFrameContext.m_lightPosition = { -1.0f, -5.0f, 0.0f};
        m_DrawFrameContext.m_lightColor = { 1.0f, 1.0f, 1.0f, 1.0f };
    }
}

void GraphicsManager::InitializeBuffers(const Scene& scene)
{
    cout << "[GraphicsManager] GraphicsManager::InitializeBuffers()" << endl;
}

void GraphicsManager::ClearBuffers()
{
    cout << "[GraphicsManager] GraphicsManager::ClearBuffers()" << endl;
}

void GraphicsManager::RenderBuffers()
{
    cout << "[GraphicsManager] GraphicsManager::RenderBuffers()" << endl;
}

#ifdef DEBUG
void GraphicsManager::RenderDebugBuffers()
{
    cout << "[GraphicsManager] GraphicsManager::RenderDebugBuffers()" << endl;
}

void GraphicsManager::DrawPoint(const Point& point, const Vector3f& color)
{
    cout << "[GraphicsManager] GraphicsManager::DrawPoint(" << point << ","
        << color << ")" << endl;
}

void GraphicsManager::DrawPointSet(const PointSet& point_set, const Vector3f& color)
{
    cout << "[GraphicsManager] GraphicsManager::DrawPointSet(" << point_set.size() << ","
        << color << ")" << endl;
}

void GraphicsManager::DrawPointSet(const PointSet& point_set, const Matrix4X4f& trans, const Vector3f& color)
{
    cout << "[GraphicsManager] GraphicsManager::DrawPointSet(" << point_set.size() << ","
        << trans << "," 
        << color << ")" << endl;
}

void GraphicsManager::DrawLine(const Point& from, const Point& to, const Vector3f& color)
{
    cout << "[GraphicsManager] GraphicsManager::DrawLine(" << from << ","
        << to << "," 
        << color << ")" << endl;
}

void GraphicsManager::DrawLine(const PointList& vertices, const Vector3f& color)
{
    cout << "[GraphicsManager] GraphicsManager::DrawLine(" << vertices.size() << ","
        << color << ")" << endl;
}

void GraphicsManager::DrawLine(const PointList& vertices, const Matrix4X4f& trans, const Vector3f& color)
{
    cout << "[GraphicsManager] GraphicsManager::DrawLine(" << vertices.size() << ","
        << trans << "," 
        << color << ")" << endl;
}

void GraphicsManager::DrawEdgeList(const EdgeList& edges, const Vector3f& color)
{
    PointList point_list;

    for (auto edge : edges)
    {
        point_list.push_back(edge->first);
        point_list.push_back(edge->second);
    }

    DrawLine(point_list, color);
}

void GraphicsManager::DrawTriangle(const PointList& vertices, const Vector3f& color)
{
    cout << "[GraphicsManager] GraphicsManager::DrawTriangle(" << vertices.size() << ","
        << color << ")" << endl;
}

void GraphicsManager::DrawTriangle(const PointList& vertices, const Matrix4X4f& trans, const Vector3f& color)
{
    cout << "[GraphicsManager] GraphicsManager::DrawTriangle(" << vertices.size() << ","
        << color << ")" << endl;
}

void GraphicsManager::DrawTriangleStrip(const PointList& vertices, const Vector3f& color)
{
    cout << "[GraphicsManager] GraphicsManager::DrawTriangleStrip(" << vertices.size() << ","
        << color << ")" << endl;
}

void GraphicsManager::DrawPolygon(const Face& polygon, const Vector3f& color)
{
    PointSet vertices;
    PointList edges;
    for (auto pEdge : polygon.Edges)
    {
        vertices.insert({pEdge->first, pEdge->second});
        edges.push_back(pEdge->first);
        edges.push_back(pEdge->second);
    }
    DrawLine(edges, color);

    DrawPointSet(vertices, color);

    DrawTriangle(polygon.GetVertices(), color * 0.5f);
}

void GraphicsManager::DrawPolygon(const Face& polygon, const Matrix4X4f& trans, const Vector3f& color)
{
    PointSet vertices;
    PointList edges;
    for (auto pEdge : polygon.Edges)
    {
        vertices.insert({pEdge->first, pEdge->second});
        edges.push_back(pEdge->first);
        edges.push_back(pEdge->second);
    }
    DrawLine(edges, trans, color);

    DrawPointSet(vertices, trans, color);

    DrawTriangle(polygon.GetVertices(), trans, color * 0.5f);
}

void GraphicsManager::DrawPolyhydron(const Polyhedron& polyhedron, const Vector3f& color)
{
    for (auto pFace : polyhedron.Faces)
    {
        DrawPolygon(*pFace, color);
    }
}

void GraphicsManager::DrawPolyhydron(const Polyhedron& polyhedron, const Matrix4X4f& trans, const Vector3f& color)
{
    for (auto pFace : polyhedron.Faces)
    {
        DrawPolygon(*pFace, trans, color);
    }
}

void GraphicsManager::DrawBox(const Vector3f& bbMin, const Vector3f& bbMax, const Vector3f& color)
{
    //  ******0--------3********
    //  *****/:       /|********
    //  ****1--------2 |********
    //  ****| :      | |********
    //  ****| 4- - - | 7********
    //  ****|/       |/*********
    //  ****5--------6**********

    // vertices
    PointPtr points[8];
    for (int i = 0; i < 8; i++)
        points[i] = make_shared<Point>(bbMin);
    *points[0] = *points[2] = *points[3] = *points[7] = bbMax;
    points[0]->x = bbMin.x;
    points[2]->y = bbMin.y;
    points[7]->z = bbMin.z;
    points[1]->z = bbMax.z;
    points[4]->y = bbMax.y;
    points[6]->x = bbMax.x;

    // edges
    EdgeList edges;
    
    // top
    edges.push_back(make_shared<Edge>(make_pair(points[0], points[3])));
    edges.push_back(make_shared<Edge>(make_pair(points[3], points[2])));
    edges.push_back(make_shared<Edge>(make_pair(points[2], points[1])));
    edges.push_back(make_shared<Edge>(make_pair(points[1], points[0])));

    // bottom
    edges.push_back(make_shared<Edge>(make_pair(points[4], points[7])));
    edges.push_back(make_shared<Edge>(make_pair(points[7], points[6])));
    edges.push_back(make_shared<Edge>(make_pair(points[6], points[5])));
    edges.push_back(make_shared<Edge>(make_pair(points[5], points[4])));

    // side
    edges.push_back(make_shared<Edge>(make_pair(points[0], points[4])));
    edges.push_back(make_shared<Edge>(make_pair(points[1], points[5])));
    edges.push_back(make_shared<Edge>(make_pair(points[2], points[6])));
    edges.push_back(make_shared<Edge>(make_pair(points[3], points[7])));

    DrawEdgeList(edges, color);
}

void GraphicsManager::ClearDebugBuffers()
{
    cout << "[GraphicsManager] GraphicsManager::ClearDebugBuffers(void)" << endl;
}
#endif

