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

    // Generate the view matrix based on the camera's position.
    CalculateCameraMatrix();
    CalculateLights();

    Clear();
    Draw();
}

void GraphicsManager::Clear()
{
}

void GraphicsManager::Draw()
{
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

void GraphicsManager::DrawLine(const Vector3f& from, const Vector3f& to, const Vector3f& color)
{
    cout << "[GraphicsManager] GraphicsManager::DrawLine(" << from << ","
        << to << "," 
        << color << ")" << endl;
}

void GraphicsManager::DrawTriangle(const PointList& vertices, const Vector3f& color)
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
    PointList _vertices;
    for (auto pEdge : polygon.Edges)
    {
        DrawLine(*pEdge->first, *pEdge->second, color);
        vertices.insert({pEdge->first, pEdge->second});
        _vertices.push_back(pEdge->first);
    }
    DrawPointSet(vertices, color);

    DrawTriangle(_vertices, color);
}

void GraphicsManager::DrawPolyhydron(const Polyhedron& polyhedron, const Vector3f& color)
{
    for (auto pFace : polyhedron.Faces)
    {
        DrawPolygon(*pFace, color);
    }
}

void GraphicsManager::DrawBox(const Vector3f& bbMin, const Vector3f& bbMax, const Vector3f& color)
{
    cout << "[GraphicsManager] GraphicsManager::DrawBox(" << bbMin << ","
        << bbMax << "," 
        << color << ")" << endl;
}

void GraphicsManager::ClearDebugBuffers()
{
    cout << "[GraphicsManager] GraphicsManager::ClearDebugBuffers(void)" << endl;
}
#endif

