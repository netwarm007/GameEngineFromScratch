#include "GraphicsManager.hpp"

#include <cstring>
#include <iostream>

#include "BRDFIntegrator.hpp"
#include "ISceneManager.hpp"
#include "SceneCamera.hpp"
#include "SceneLighting.hpp"

#include "ForwardGeometryPass.hpp"
#include "ShadowMapPass.hpp"
#include "OverlayPass.hpp"

#include "imgui.h"

using namespace My;
using namespace std;

GraphicsManager::GraphicsManager() {
    m_Frames.resize(GfxConfiguration::kMaxInFlightFrameCount);
}

GraphicsManager::GraphicsManager(ISceneManager& scenes,
                                 IPipelineStateManager& pipelines)
    : GraphicsManager() {
    m_pSceneManager = &scenes;
    m_pPipelineStateManager = &pipelines;
}

int GraphicsManager::Initialize() {
    if (!m_pApp) {
        cerr << "[GraphicsManager] Application must be set before initialization"
             << endl;
        return -1;
    }

    m_pApp->GetFramebufferSize(m_canvasWidth, m_canvasHeight);

    auto* pPipelineStateMgr = m_pPipelineStateManager;
    if (pPipelineStateMgr) {
#if !defined(OS_WEBASSEMBLY)
        m_InitPasses.push_back(
            make_shared<BRDFIntegrator>(this, pPipelineStateMgr));
#endif
        // m_DispatchPasses.push_back(make_shared<RayTracePass>(this,
        // pPipelineStateMgr));
        m_DrawPasses.push_back(
            make_shared<ShadowMapPass>(this, pPipelineStateMgr));

        auto forward_pass = make_shared<ForwardGeometryPass>(this, pPipelineStateMgr);

        forward_pass->EnableRenderToTexture();

        m_DrawPasses.push_back(forward_pass);
        m_DrawPasses.push_back(
            make_shared<OverlayPass>(this, pPipelineStateMgr));
    }

    InitConstants();

    m_bInitialize = true;

    return 0;
}

void GraphicsManager::Finalize() {
    EndScene();
    m_bInitialize = false;
}

void GraphicsManager::Tick() {
    auto* pSceneManager = m_pSceneManager;
    if (pSceneManager) {
        auto rev = pSceneManager->GetSceneRevision();
        if (rev == 0) return;  // scene is not loaded yet
        const auto scene = pSceneManager->GetSceneForRendering();
        if (!scene) {
            cerr << "[GraphicsManager] Scene revision has no rendering snapshot"
                 << endl;
            return;
        }
        assert(m_nSceneRevision <= rev);
        if (m_nSceneRevision < rev) {
            EndScene();
            cerr << "[GraphicsManager] Detected Scene Change, reinitialize "
                    "buffers "
                    "..."
                 << endl;
            BeginScene(*scene);
            m_nSceneRevision = rev;
        }
        m_Frames[m_nFrameIndex].scene = scene;
    }

    if (!UpdateConstants()) return;

    BeginFrame(m_Frames[m_nFrameIndex]);
    ImGui::NewFrame();
    Draw();
    ImGui::EndFrame();
    ImGui::Render();
    EndFrame(m_Frames[m_nFrameIndex]);

    Present();

    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
    }
}

void GraphicsManager::createFramebuffers() {
    assert(m_pApp);
    auto conf = m_pApp->GetConfiguration();

    releaseFramebuffers();

    for (int32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        auto& color_texture = m_ColorTextures[i];
        color_texture.width = m_canvasWidth;
        color_texture.height = m_canvasHeight;
        color_texture.mips = 1;
        color_texture.pixel_format = PIXEL_FORMAT::RGBA8;
        color_texture.samples = 1;

        GenerateTexture(color_texture);

        m_Frames[i].colorTextures.push_back(color_texture);

        if (i == 0) {
            // Generate msaa intermediate RT
            if (conf.msaaSamples > 1) {
                m_MsaaColorTexture = color_texture;
                m_MsaaColorTexture.handler = 0;
                m_MsaaColorTexture.buff = 0;
                m_MsaaColorTexture.samples = conf.msaaSamples;
                GenerateTexture(m_MsaaColorTexture);
            }

            // Generate depth RT
            auto& depth_buffer = m_DepthTexture;
            depth_buffer.width = m_canvasWidth;
            depth_buffer.height = m_canvasHeight;
            depth_buffer.mips = 1;
            depth_buffer.pixel_format = PIXEL_FORMAT::D32;
            depth_buffer.samples = conf.msaaSamples;

            GenerateTexture(depth_buffer);
        }

        if (conf.msaaSamples > 1) {
            m_Frames[i].colorTextures.push_back(m_MsaaColorTexture);
            m_Frames[i].enableMSAA = true;
        }
        m_Frames[i].depthTexture = m_DepthTexture;
    }
}

void GraphicsManager::releaseFramebuffers() {
    for (auto& frame : m_Frames) {
        frame.colorTextures.clear();
        frame.depthTexture = {};
        frame.enableMSAA = false;
    }

    auto release = [this](Texture2D& texture) {
        if (texture.handler || texture.buff) {
            ReleaseTexture(texture);
        }
        texture = {};
    };
    for (auto& texture : m_ColorTextures) {
        release(texture);
    }
    release(m_MsaaColorTexture);
    release(m_DepthTexture);
}

void GraphicsManager::ResizeCanvas(int32_t width, int32_t height) {
    if (width > 0 && height > 0 && (m_canvasWidth != width || m_canvasHeight != height)) {
        cerr << "[GraphicsManager] Resize Canvas to " << width << "x" << height
            << endl;
        m_canvasWidth = width;
        m_canvasHeight = height;

        if (m_bInitialize) {
            // resize frame buffers (for offline-rendering)
            createFramebuffers();
        }
    }
}

bool GraphicsManager::UpdateConstants() {
    // update scene object position
    auto& frame = m_Frames[m_nFrameIndex];

    for (auto& pDbc : frame.batchContexts) {
        pDbc->modelMatrix = pDbc->node->GetRenderTransform();
    }

    // Generate the view matrix based on the camera's position.
    if (!CalculateCameraMatrix()) return false;
    return CalculateLights();
}

void GraphicsManager::Draw() {
    auto& frame = m_Frames[m_nFrameIndex];

    for (auto& pDispatchPass : m_DispatchPasses) {
        pDispatchPass->BeginPass(frame);
        pDispatchPass->Dispatch(frame);
        pDispatchPass->EndPass(frame);
    }

    for (auto& pDrawPass : m_DrawPasses) {
        pDrawPass->BeginPass(frame);
        pDrawPass->Draw(frame);
        pDrawPass->EndPass(frame);
    }
}

bool GraphicsManager::CalculateCameraMatrix() {
    auto& frame = m_Frames[m_nFrameIndex];
    if (!frame.scene) return true;

    CameraFrame camera;
    if (!PrepareSceneCamera(*frame.scene, m_canvasWidth, m_canvasHeight,
                            m_pApp->GetConfiguration().fixOpenGLPerspectiveMatrix,
                            camera)) {
        return false;
    }
    frame.frameContext.viewMatrix = camera.viewMatrix;
    frame.frameContext.projectionMatrix = camera.projectionMatrix;
    frame.frameContext.camPos = camera.position;
    return true;
}

bool GraphicsManager::CalculateLights() {
    auto& frame = m_Frames[m_nFrameIndex];
    const bool openGL = m_pApp->GetConfiguration().fixOpenGLPerspectiveMatrix;
    LightingFrame lighting;
    lighting.clipSpaceType = openGL ? 0 : 1;
    if (frame.scene && !PrepareSceneLights(*frame.scene, openGL, lighting)) {
        return false;
    }
    frame.lightInfo = lighting.lightInfo;
    frame.frameContext.numLights = lighting.numLights;
    frame.frameContext.clip_space_type = lighting.clipSpaceType;
    return true;
}

void GraphicsManager::BeginScene(const Scene& scene) {
    // first, call init passes on frame 0
    for (const auto& pPass : m_InitPasses) {
        pPass->BeginPass(m_Frames[0]);
        pPass->Dispatch(m_Frames[0]);
        pPass->EndPass(m_Frames[0]);
    }

    // generate shadow map array
    if (!m_Frames[0].frameContext.shadowMap.handler) {
        m_Frames[0].frameContext.shadowMap.width = GfxConfiguration::kShadowMapWidth;
        m_Frames[0].frameContext.shadowMap.height = GfxConfiguration::kShadowMapHeight;
        m_Frames[0].frameContext.shadowMap.size = GfxConfiguration::kMaxShadowMapCount;
        m_Frames[0].frameContext.shadowMap.pixel_format = PIXEL_FORMAT::D32;
        GenerateTextureArray(m_Frames[0].frameContext.shadowMap);
    }

    // generate global shadow map array
    if (!m_Frames[0].frameContext.globalShadowMap.handler) {
        m_Frames[0].frameContext.globalShadowMap.width = GfxConfiguration::kShadowMapWidth;
        m_Frames[0].frameContext.globalShadowMap.height = GfxConfiguration::kShadowMapHeight;
        m_Frames[0].frameContext.globalShadowMap.size = GfxConfiguration::kMaxShadowMapCount;
        m_Frames[0].frameContext.globalShadowMap.pixel_format = PIXEL_FORMAT::D32;
        GenerateTextureArray(m_Frames[0].frameContext.globalShadowMap);
    }

    // generate cube shadow map array
    if (!m_Frames[0].frameContext.cubeShadowMap.handler) {
        m_Frames[0].frameContext.cubeShadowMap.width = GfxConfiguration::kShadowMapWidth;
        m_Frames[0].frameContext.cubeShadowMap.height = GfxConfiguration::kShadowMapHeight;
        m_Frames[0].frameContext.cubeShadowMap.size = GfxConfiguration::kMaxShadowMapCount;
        m_Frames[0].frameContext.cubeShadowMap.pixel_format = PIXEL_FORMAT::D32;
        GenerateCubemapArray(m_Frames[0].frameContext.cubeShadowMap);
    }

    if (scene.Geometries.size()) {
        initializeGeometries(scene);
    }
    if (scene.SkyBox) {
        initializeSkyBox(scene);
    }

    // now, copy the frame structures and initialize shadow maps
    for (int32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        m_Frames[i] = m_Frames[0];
        m_Frames[i].frameIndex = i;
    }

    // generate frame buffers (for offline-rendering)
    // please note we must put this after the copy above to avoid overwritten
    createFramebuffers();
}

void GraphicsManager::EndScene() {
    for (auto& frame : m_Frames) {
        frame.batchContexts.clear();
        frame.brdfLUT = {};
        frame.skybox = {};
        frame.frameContext.shadowMap = {};
        frame.frameContext.globalShadowMap = {};
        frame.frameContext.cubeShadowMap = {};
        frame.scene.reset();
    }
    material_map.clear();
    for (auto& texture : m_Textures) {
        ReleaseTexture(texture);
    }
    m_Textures.clear();
    releaseFramebuffers();
}
