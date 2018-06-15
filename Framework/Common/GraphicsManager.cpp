#include <iostream>
#include "GraphicsManager.hpp"
#include "SceneManager.hpp"
#include "cbuffer.h"
#include "IApplication.hpp"
#include "ForwardRenderPass.hpp"

using namespace My;
using namespace std;

int GraphicsManager::Initialize()
{
    int result = 0;
    m_Frames.resize(kFrameCount);
	InitConstants();
    m_pBasePass = make_shared<ForwardRenderPass>();
    return result;
}

void GraphicsManager::Finalize()
{
#ifdef DEBUG
    ClearDebugBuffers();
#endif
    ClearBuffers();
}

void GraphicsManager::Tick()
{
    if (g_pSceneManager->IsSceneChanged())
    {
        cout << "[GraphicsManager] Detected Scene Change, reinitialize buffers ..." << endl;
        ClearBuffers();
        const Scene& scene = g_pSceneManager->GetSceneForRendering();
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

    if (m_pBasePass)
    {
        m_pBasePass->Draw(m_Frames[m_nFrameIndex]);
    }

#ifdef DEBUG
    RenderDebugBuffers();
#endif
}

void GraphicsManager::InitConstants()
{
    // Initialize the world/model matrix to the identity matrix.
    BuildIdentityMatrix(m_Frames[m_nFrameIndex].frameContext.m_worldMatrix);
}

void GraphicsManager::CalculateCameraMatrix()
{
    auto& scene = g_pSceneManager->GetSceneForRendering();
    auto pCameraNode = scene.GetFirstCameraNode();
    DrawFrameContext& frameContext = m_Frames[m_nFrameIndex].frameContext;
    if (pCameraNode) {
        auto transform = *pCameraNode->GetCalculatedTransform();
        InverseMatrix4X4f(transform);
        frameContext.m_viewMatrix = transform;
    }
    else {
        // use default build-in camera
        Vector3f position = { 0.0f, -5.0f, 0.0f }, lookAt = { 0.0f, 0.0f, 0.0f }, up = { 0.0f, 0.0f, 1.0f };
        BuildViewMatrix(frameContext.m_viewMatrix, position, lookAt, up);
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
    BuildPerspectiveFovRHMatrix(frameContext.m_projectionMatrix, fieldOfView, screenAspect, nearClipDistance, farClipDistance);
}

void GraphicsManager::CalculateLights()
{
    DrawFrameContext& frameContext = m_Frames[m_nFrameIndex].frameContext;
    frameContext.m_ambientColor = { 0.01f, 0.01f, 0.01f };
    frameContext.m_lights.clear();

    auto& scene = g_pSceneManager->GetSceneForRendering();
    for (auto LightNode : scene.LightNodes) {
        Light light;
        auto pLightNode = LightNode.second.lock();
        if (!pLightNode) continue;
        auto trans_ptr = pLightNode->GetCalculatedTransform();
        Transform(light.m_lightPosition, *trans_ptr);
        Transform(light.m_lightDirection, *trans_ptr);

        auto pLight = scene.GetLight(pLightNode->GetSceneObjectRef());
        if (pLight) {
            light.m_lightColor = pLight->GetColor().Value;
            light.m_lightIntensity = pLight->GetIntensity();
            const AttenCurve& atten_curve = pLight->GetDistanceAttenuation();
            light.m_lightDistAttenCurveType = atten_curve.type; 
            memcpy(light.m_lightDistAttenCurveParams, &atten_curve.u, sizeof(atten_curve.u));

            if (pLight->GetType() == SceneObjectType::kSceneObjectTypeLightInfi)
            {
                light.m_lightPosition[3] = 0.0f;
            }
            else if (pLight->GetType() == SceneObjectType::kSceneObjectTypeLightSpot)
            {
                auto plight = dynamic_pointer_cast<SceneObjectSpotLight>(pLight);
                const AttenCurve& angle_atten_curve = plight->GetAngleAttenuation();
                light.m_lightAngleAttenCurveType = angle_atten_curve.type;
                memcpy(light.m_lightAngleAttenCurveParams, &angle_atten_curve.u, sizeof(angle_atten_curve.u));
            }
            else if (pLight->GetType() == SceneObjectType::kSceneObjectTypeLightArea)
            {
                auto plight = dynamic_pointer_cast<SceneObjectAreaLight>(pLight);
                light.m_lightSize = plight->GetDimension();
            }
        }
        else
        {
            assert(0);
        }

        frameContext.m_lights.push_back(light);
    }
}

void GraphicsManager::InitializeBuffers(const Scene& scene)
{
    cout << "[GraphicsManager] InitializeBuffers()" << endl;
}

void GraphicsManager::ClearBuffers()
{
    cout << "[GraphicsManager] ClearBuffers()" << endl;
}

#ifdef DEBUG
void GraphicsManager::RenderDebugBuffers()
{
    cout << "[GraphicsManager] RenderDebugBuffers()" << endl;
}

void GraphicsManager::DrawPoint(const Point& point, const Vector3f& color)
{
    cout << "[GraphicsManager] DrawPoint(" << point << ","
        << color << ")" << endl;
}

void GraphicsManager::DrawPointSet(const PointSet& point_set, const Vector3f& color)
{
    cout << "[GraphicsManager] DrawPointSet(" << point_set.size() << ","
        << color << ")" << endl;
}

void GraphicsManager::DrawPointSet(const PointSet& point_set, const Matrix4X4f& trans, const Vector3f& color)
{
    cout << "[GraphicsManager] DrawPointSet(" << point_set.size() << ","
        << trans << "," 
        << color << ")" << endl;
}

void GraphicsManager::DrawLine(const Point& from, const Point& to, const Vector3f& color)
{
    cout << "[GraphicsManager] DrawLine(" << from << ","
        << to << "," 
        << color << ")" << endl;
}

void GraphicsManager::DrawLine(const PointList& vertices, const Vector3f& color)
{
    cout << "[GraphicsManager] DrawLine(" << vertices.size() << ","
        << color << ")" << endl;
}

void GraphicsManager::DrawLine(const PointList& vertices, const Matrix4X4f& trans, const Vector3f& color)
{
    cout << "[GraphicsManager] DrawLine(" << vertices.size() << ","
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
    cout << "[GraphicsManager] DrawTriangle(" << vertices.size() << ","
        << color << ")" << endl;
}

void GraphicsManager::DrawTriangle(const PointList& vertices, const Matrix4X4f& trans, const Vector3f& color)
{
    cout << "[GraphicsManager] DrawTriangle(" << vertices.size() << ","
        << color << ")" << endl;
}

void GraphicsManager::DrawTriangleStrip(const PointList& vertices, const Vector3f& color)
{
    cout << "[GraphicsManager] DrawTriangleStrip(" << vertices.size() << ","
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
    points[0]->data[0] = bbMin[0];
    points[2]->data[1] = bbMin[1];
    points[7]->data[2] = bbMin[2];
    points[1]->data[2] = bbMax[2];
    points[4]->data[1] = bbMax[1];
    points[6]->data[0] = bbMax[0];

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
    cout << "[GraphicsManager] ClearDebugBuffers(void)" << endl;
}
#endif

void GraphicsManager::SetBasePass(shared_ptr<IDrawPass> pBasePass)
{
    m_pBasePass = pBasePass;
}

void GraphicsManager::UseShaderProgram(void* shaderProgram)
{
    cout << "[GraphicsManager] UseShaderProgram(" << shaderProgram << ")" << endl;
}

void GraphicsManager::SetPerFrameConstants(const DrawFrameContext& context)
{
    cout << "[GraphicsManager] SetPerFrameConstants(" << &context << ")" << endl;
}

void GraphicsManager::DrawBatch(const DrawBatchContext& context)
{
    cout << "[GraphicsManager] DrawBatch(" << &context << ")" << endl;
}
