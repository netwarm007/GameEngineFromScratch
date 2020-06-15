#include "GraphicsManager.hpp"

#include <cstring>
#include <iostream>

#include "BRDFIntegrator.hpp"
#include "ForwardGeometryPass.hpp"
#include "IApplication.hpp"
#include "IPhysicsManager.hpp"
#include "RayTracePass.hpp"
#include "SceneManager.hpp"
#include "ShadowMapPass.hpp"

using namespace My;
using namespace std;

int GraphicsManager::Initialize() {
    int result = 0;
#if !defined(OS_WEBASSEMBLY)
    m_InitPasses.push_back(make_shared<BRDFIntegrator>());
#endif
    //m_DispatchPasses.push_back(make_shared<RayTracePass>());
    m_DrawPasses.push_back(make_shared<ShadowMapPass>());
    m_DrawPasses.push_back(make_shared<ForwardGeometryPass>());

    InitConstants();
    return result;
}

void GraphicsManager::Finalize() {
#ifdef DEBUG
    ClearDebugBuffers();
#endif
    EndScene();
}

void GraphicsManager::Tick() {
    auto rev = g_pSceneManager->GetSceneRevision();
    if (m_nSceneRevision != rev) {
        EndScene();
        cerr << "[GraphicsManager] Detected Scene Change, reinitialize buffers "
                "..."
             << endl;
        const auto scene = g_pSceneManager->GetSceneForRendering();
        BeginScene(*scene);
        m_nSceneRevision = rev;
    }

    UpdateConstants();

    BeginFrame(m_Frames[m_nFrameIndex]);
    Draw();
    EndFrame(m_Frames[m_nFrameIndex]);

    Present();
}

void GraphicsManager::ResizeCanvas(int32_t width, int32_t height) {
    cerr << "[GraphicsManager] Resize Canvas to " << width << "x" << height
         << endl;
}

void GraphicsManager::UpdateConstants() {
    // update scene object position
    auto& frame = m_Frames[m_nFrameIndex];

    for (auto& pDbc : frame.batchContexts) {
        if (void* rigidBody = pDbc->node->RigidBody()) {
            Matrix4X4f trans;

            // the geometry has rigid body bounded, we blend the simlation
            // result here.
            Matrix4X4f simulated_result =
                g_pPhysicsManager->GetRigidBodyTransform(rigidBody);

            BuildIdentityMatrix(trans);

            // apply the rotation part of the simlation result
            memcpy(trans[0], simulated_result[0], sizeof(float) * 3);
            memcpy(trans[1], simulated_result[1], sizeof(float) * 3);
            memcpy(trans[2], simulated_result[2], sizeof(float) * 3);

            // replace the translation part of the matrix with simlation result
            // directly
            memcpy(trans[3], simulated_result[3], sizeof(float) * 3);

            pDbc->modelMatrix = trans;
        } else {
            pDbc->modelMatrix = *pDbc->node->GetCalculatedTransform();
        }
    }

    // Generate the view matrix based on the camera's position.
    CalculateCameraMatrix();
    CalculateLights();
}

void GraphicsManager::Draw() {
    auto& frame = m_Frames[m_nFrameIndex];

    for (auto& pDispatchPass : m_DispatchPasses) {
        pDispatchPass->BeginPass();
        pDispatchPass->Dispatch(frame);
        pDispatchPass->EndPass();
    }

    for (auto& pDrawPass : m_DrawPasses) {
        pDrawPass->BeginPass();
        pDrawPass->Draw(frame);
        pDrawPass->EndPass();
    }
}

void GraphicsManager::CalculateCameraMatrix() {
    auto& scene = g_pSceneManager->GetSceneForRendering();
    auto pCameraNode = scene->GetFirstCameraNode();
    DrawFrameContext& frameContext = m_Frames[m_nFrameIndex].frameContext;
    if (pCameraNode) {
        auto transform = *pCameraNode->GetCalculatedTransform();
        frameContext.camPos =
            Vector3f({transform[3][0], transform[3][1], transform[3][2]});
        InverseMatrix4X4f(transform);
        frameContext.viewMatrix = transform;
    } else {
        // use default build-in camera
        Vector3f position = {0.0f, -5.0f, 0.0f}, lookAt = {0.0f, 0.0f, 0.0f},
                 up = {0.0f, 0.0f, 1.0f};
        BuildViewRHMatrix(frameContext.viewMatrix, position, lookAt, up);
    }

    float fieldOfView = PI / 3.0f;
    float nearClipDistance = 1.0f;
    float farClipDistance = 100.0f;

    if (pCameraNode) {
        auto pCamera = scene->GetCamera(pCameraNode->GetSceneObjectRef());
        // Set the field of view and screen aspect ratio.
        fieldOfView =
            dynamic_pointer_cast<SceneObjectPerspectiveCamera>(pCamera)
                ->GetFov();
        nearClipDistance = pCamera->GetNearClipDistance();
        farClipDistance = pCamera->GetFarClipDistance();
    }

    const GfxConfiguration& conf = g_pApp->GetConfiguration();

    float screenAspect = (float)conf.screenWidth / (float)conf.screenHeight;

    // Build the perspective projection matrix.
    BuildPerspectiveFovRHMatrix(frameContext.projectionMatrix, fieldOfView,
                                screenAspect, nearClipDistance,
                                farClipDistance);
}

void GraphicsManager::CalculateLights() {
    DrawFrameContext& frameContext = m_Frames[m_nFrameIndex].frameContext;
    auto& light_info = m_Frames[m_nFrameIndex].lightInfo;

    frameContext.numLights = 0;

    auto& scene = g_pSceneManager->GetSceneForRendering();
    for (const auto& LightNode : scene->LightNodes) {
        Light& light = light_info.lights[frameContext.numLights];
        auto pLightNode = LightNode.second.lock();
        if (!pLightNode) continue;
        auto trans_ptr = pLightNode->GetCalculatedTransform();
        light.lightPosition = {0.0f, 0.0f, 0.0f, 1.0f};
        light.lightDirection = {0.0f, 0.0f, -1.0f, 0.0f};
        Transform(light.lightPosition, *trans_ptr);
        Transform(light.lightDirection, *trans_ptr);
        Normalize(light.lightDirection);

        auto pLight = scene->GetLight(pLightNode->GetSceneObjectRef());
        if (pLight) {
            light.lightGuid = pLight->GetGuid();
            light.lightColor = pLight->GetColor().Value;
            light.lightIntensity = pLight->GetIntensity();
            light.lightCastShadow = pLight->GetIfCastShadow();
            const AttenCurve& atten_curve = pLight->GetDistanceAttenuation();
            light.lightDistAttenCurveType = atten_curve.type;
            memcpy(light.lightDistAttenCurveParams, &atten_curve.u,
                   sizeof(atten_curve.u));
            light.lightAngleAttenCurveType = AttenCurveType::kNone;

            Matrix4X4f view;
            Matrix4X4f projection;
            BuildIdentityMatrix(projection);

            float nearClipDistance = 1.0f;
            float farClipDistance = 1000.0f;

            if (pLight->GetType() ==
                SceneObjectType::kSceneObjectTypeLightInfi) {
                light.lightType = LightType::Infinity;

                Vector4f target = {0.0f, 0.0f, 0.0f, 1.0f};

                auto pCameraNode = scene->GetFirstCameraNode();
                if (pCameraNode) {
                    auto pCamera =
                        scene->GetCamera(pCameraNode->GetSceneObjectRef());
                    nearClipDistance = pCamera->GetNearClipDistance();
                    farClipDistance = pCamera->GetFarClipDistance();

                    target[2] =
                        -(0.75f * nearClipDistance + 0.25f * farClipDistance);

                    // calculate the camera target position
                    auto trans_ptr = pCameraNode->GetCalculatedTransform();
                    Transform(target, *trans_ptr);
                }

                light.lightPosition =
                    target - light.lightDirection * farClipDistance;
                Vector3f position;
                position.Set((float*)light.lightPosition);
                Vector3f lookAt;
                lookAt.Set((float*)target);
                Vector3f up = {0.0f, 0.0f, 1.0f};
                if (abs(light.lightDirection[0]) <= 0.2f &&
                    abs(light.lightDirection[1]) <= 0.2f) {
                    up = {0.1f, 0.1f, 1.0f};
                }
                BuildViewRHMatrix(view, position, lookAt, up);

                float sm_half_dist = min(farClipDistance * 0.25f, 800.0f);

                BuildOrthographicMatrix(projection, -sm_half_dist, sm_half_dist,
                                        sm_half_dist, -sm_half_dist,
                                        nearClipDistance,
                                        farClipDistance + sm_half_dist);

                // notify shader about the infinity light by setting 4th field
                // to 0
                light.lightPosition[3] = 0.0f;
            } else {
                Vector3f position;
                position.Set(light.lightPosition);
                Vector4f tmp = light.lightPosition + light.lightDirection;
                Vector3f lookAt;
                lookAt.Set(tmp);
                Vector3f up = {0.0f, 0.0f, 1.0f};
                if (abs(light.lightDirection[0]) <= 0.1f &&
                    abs(light.lightDirection[1]) <= 0.1f) {
                    up = {0.0f, 0.707f, 0.707f};
                }
                BuildViewRHMatrix(view, position, lookAt, up);

                if (pLight->GetType() ==
                    SceneObjectType::kSceneObjectTypeLightSpot) {
                    light.lightType = LightType::Spot;

                    auto plight =
                        dynamic_pointer_cast<SceneObjectSpotLight>(pLight);
                    const AttenCurve& angle_atten_curve =
                        plight->GetAngleAttenuation();
                    light.lightAngleAttenCurveType = angle_atten_curve.type;
                    memcpy(light.lightAngleAttenCurveParams,
                           &angle_atten_curve.u, sizeof(angle_atten_curve.u));

                    float fieldOfView =
                        light.lightAngleAttenCurveParams[0][1] * 2.0f;
                    float screenAspect = 1.0f;

                    // Build the perspective projection matrix.
                    BuildPerspectiveFovRHMatrix(projection, fieldOfView,
                                                screenAspect, nearClipDistance,
                                                farClipDistance);
                } else if (pLight->GetType() ==
                           SceneObjectType::kSceneObjectTypeLightArea) {
                    light.lightType = LightType::Area;

                    auto plight =
                        dynamic_pointer_cast<SceneObjectAreaLight>(pLight);
                    light.lightSize = plight->GetDimension();
                } else  // omni light
                {
                    light.lightType = LightType::Omni;

                    // auto plight =
                    // dynamic_pointer_cast<SceneObjectOmniLight>(pLight);

                    float fieldOfView =
                        PI / 2.0f;  // 90 degree for each cube map face
                    float screenAspect = 1.0f;

                    // Build the perspective projection matrix.
                    BuildPerspectiveFovRHMatrix(projection, fieldOfView,
                                                screenAspect, nearClipDistance,
                                                farClipDistance);
                }
            }

            light.lightViewMatrix = view;
            light.lightProjectionMatrix = projection;
            frameContext.numLights++;
        } else {
            assert(0);
        }
    }
}

void GraphicsManager::BeginScene(const Scene& scene) {
    // first, call init passes on frame 0
    for (const auto& pPass : m_InitPasses) {
        pPass->BeginPass();
        pPass->Dispatch(m_Frames[0]);
        pPass->EndPass();
    }

    // now, copy the frame structures and initialize shadow maps
    for (int32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        m_Frames[i] = m_Frames[0];
        m_Frames[i].frameIndex = i;

        // generate shadow map array
        if (m_Frames[i].frameContext.shadowMap == -1) {
            m_Frames[i].frameContext.shadowMap =
                g_pGraphicsManager->GenerateShadowMapArray(
                    GfxConfiguration::kShadowMapWidth,
                    GfxConfiguration::kShadowMapHeight,
                    GfxConfiguration::kMaxShadowMapCount);
        }

        // generate global shadow map array
        if (m_Frames[i].frameContext.globalShadowMap == -1) {
            m_Frames[i].frameContext.globalShadowMap =
                g_pGraphicsManager->GenerateShadowMapArray(
                    GfxConfiguration::kGlobalShadowMapWidth,
                    GfxConfiguration::kGlobalShadowMapHeight,
                    GfxConfiguration::kMaxGlobalShadowMapCount);
        }

        // generate cube shadow map array
        if (m_Frames[i].frameContext.cubeShadowMap == -1) {
            m_Frames[i].frameContext.cubeShadowMap =
                g_pGraphicsManager->GenerateCubeShadowMapArray(
                    GfxConfiguration::kCubeShadowMapWidth,
                    GfxConfiguration::kCubeShadowMapHeight,
                    GfxConfiguration::kMaxCubeShadowMapCount);
        }
    }

    if (scene.Geometries.size()) {
        initializeGeometries(scene);
    }
    if (scene.Terrain) {
        initializeTerrain(scene);
    }
    if (scene.SkyBox) {
        initializeSkyBox(scene);
    }
}

void GraphicsManager::EndScene() {}

void GraphicsManager::BeginFrame(const Frame& frame) {}

void GraphicsManager::EndFrame(const Frame&) {
    m_nFrameIndex =
        ((m_nFrameIndex + 1) % GfxConfiguration::kMaxInFlightFrameCount);
}

int32_t GraphicsManager::GetTexture(const char* id) {
    int32_t result = -1;

    auto it = m_Textures.find(id);
    if (it != m_Textures.end()) {
        result = it->second;
    }

    return result;
}

#ifdef DEBUG
void GraphicsManager::DrawEdgeList(const EdgeList& edges,
                                   const Vector3f& color) {
    PointList point_list;

    for (const auto& edge : edges) {
        point_list.push_back(edge->first);
        point_list.push_back(edge->second);
    }

    DrawLine(point_list, color);
}

void GraphicsManager::DrawPolygon(const Face& polygon, const Vector3f& color) {
    PointSet vertices;
    PointList edges;
    for (const auto& pEdge : polygon.Edges) {
        vertices.insert({pEdge->first, pEdge->second});
        edges.push_back(pEdge->first);
        edges.push_back(pEdge->second);
    }
    DrawLine(edges, color);

    DrawPointSet(vertices, color);

    DrawTriangle(polygon.GetVertices(), color * 0.5f);
}

void GraphicsManager::DrawPolygon(const Face& polygon, const Matrix4X4f& trans,
                                  const Vector3f& color) {
    PointSet vertices;
    PointList edges;
    for (const auto& pEdge : polygon.Edges) {
        vertices.insert({pEdge->first, pEdge->second});
        edges.push_back(pEdge->first);
        edges.push_back(pEdge->second);
    }
    DrawLine(edges, trans, color);

    DrawPointSet(vertices, trans, color);

    DrawTriangle(polygon.GetVertices(), trans, color * 0.5f);
}

void GraphicsManager::DrawPolyhydron(const Polyhedron& polyhedron,
                                     const Vector3f& color) {
    for (const auto& pFace : polyhedron.Faces) {
        DrawPolygon(*pFace, color);
    }
}

void GraphicsManager::DrawPolyhydron(const Polyhedron& polyhedron,
                                     const Matrix4X4f& trans,
                                     const Vector3f& color) {
    for (const auto& pFace : polyhedron.Faces) {
        DrawPolygon(*pFace, trans, color);
    }
}

void GraphicsManager::DrawBox(const Vector3f& bbMin, const Vector3f& bbMax,
                              const Vector3f& color) {
    //  ******0--------3********
    //  *****/:       /|********
    //  ****1--------2 |********
    //  ****| :      | |********
    //  ****| 4- - - | 7********
    //  ****|/       |/*********
    //  ****5--------6**********

    // vertices
    PointPtr points[8];
    for (auto& point : points) point = make_shared<Point>(bbMin);
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

#endif
