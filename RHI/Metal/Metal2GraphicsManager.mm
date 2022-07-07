#import <Metal/Metal.h>

#include "Metal2Renderer.h"
#include "MetalView.h"
#include "Metal2GraphicsManager.h"

using namespace My;
using namespace std;

int Metal2GraphicsManager::Initialize() {
    int result;

    result = GraphicsManager::Initialize();

    NSWindow* pWindow = (NSWindow*)m_pApp->GetMainWindowHandler();
    MetalView* view = [pWindow contentView];
    m_pRenderer = [[Metal2Renderer new] initWithMetalKitView:view device:view.device];

    if (result == 0) {
        [m_pRenderer initialize];
    }

    return result;
}

void Metal2GraphicsManager::Finalize() {
    [m_pRenderer finalize];
    GraphicsManager::Finalize();
}

void Metal2GraphicsManager::Present() { [m_pRenderer present]; }

void Metal2GraphicsManager::initializeGeometries(const Scene& scene) {
    uint32_t batch_index = 0;
    uint32_t v_property_offset = 0;
    uint32_t index_offset = 0;

    // load geometries (and materials)
    // TODO: load materials async
    for (auto& _it : scene.GeometryNodes) {
        auto pGeometryNode = _it.second.lock();

        if (pGeometryNode && pGeometryNode->Visible()) {
            auto pGeometry = scene.GetGeometry(pGeometryNode->GetSceneObjectRef());
            assert(pGeometry);
            auto pMesh = pGeometry->GetMesh().lock();
            if (!pMesh) continue;

            // Set the number of vertex properties.
            auto vertexPropertiesCount = pMesh->GetVertexPropertiesCount();

            // Set the number of vertices in the vertex array.
            // auto vertexCount = pMesh->GetVertexCount();

            for (decltype(vertexPropertiesCount) i = 0; i < vertexPropertiesCount; i++) {
                const SceneObjectVertexArray& v_property_array = pMesh->GetVertexPropertyArray(i);

                [m_pRenderer createVertexBuffer:v_property_array];
            }

            const SceneObjectIndexArray& index_array = pMesh->GetIndexArray(0);
            [m_pRenderer createIndexBuffer:index_array];

            MTLPrimitiveType mode;
            switch (pMesh->GetPrimitiveType()) {
                case PrimitiveType::kPrimitiveTypePointList:
                    mode = MTLPrimitiveTypePoint;
                    break;
                case PrimitiveType::kPrimitiveTypeLineList:
                    mode = MTLPrimitiveTypeLine;
                    break;
                case PrimitiveType::kPrimitiveTypeLineStrip:
                    mode = MTLPrimitiveTypeLineStrip;
                    break;
                case PrimitiveType::kPrimitiveTypeTriList:
                    mode = MTLPrimitiveTypeTriangle;
                    break;
                case PrimitiveType::kPrimitiveTypeTriStrip:
                    mode = MTLPrimitiveTypeTriangleStrip;
                    break;
                default:
                    // ignore
                    continue;
            }

            MTLIndexType type;
            switch (index_array.GetIndexType()) {
                case IndexDataType::kIndexDataTypeInt8:
                    // not supported
                    assert(0);
                    break;
                case IndexDataType::kIndexDataTypeInt16:
                    type = MTLIndexTypeUInt16;
                    break;
                case IndexDataType::kIndexDataTypeInt32:
                    type = MTLIndexTypeUInt32;
                    break;
                default:
                    // not supported by OpenGL
                    cerr << "Error: Unsupported Index Type " << index_array << endl;
                    cerr << "Mesh: " << *pMesh << endl;
                    cerr << "Geometry: " << *pGeometry << endl;
                    continue;
            }

            auto material_index = index_array.GetMaterialIndex();
            auto material_key = pGeometryNode->GetMaterialRef(material_index);
            auto material = scene.GetMaterial(material_key);

            auto dbc = make_shared<MtlDrawBatchContext>();
            dbc->batchIndex = batch_index++;
            dbc->index_offset = index_offset++;
            dbc->index_count = (uint32_t)index_array.GetIndexCount();
            dbc->index_mode = mode;
            dbc->index_type = type;
            dbc->property_offset = v_property_offset;
            dbc->property_count = vertexPropertiesCount;

            auto it = material_map.find(material_key);
            if (it == material_map.end()) {
                // load material textures
                if (material) {
                    if (auto& texture = material->GetBaseColor().ValueMap) {
                        auto image = texture->GetTextureImage();
                        if (image) {
                            adjust_image(*image);
                            dbc->material.diffuseMap.handler =
                                (TextureHandler)[m_pRenderer createTexture:*image];
                            dbc->material.diffuseMap.width = image->Width;
                            dbc->material.diffuseMap.height = image->Height;
                        }
                    }

                    if (auto& texture = material->GetNormal().ValueMap) {
                        auto image = texture->GetTextureImage();
                        if (image) {
                            adjust_image(*image);
                            dbc->material.normalMap.handler =
                                (TextureHandler)[m_pRenderer createTexture:*image];
                            dbc->material.normalMap.width = image->Width;
                            dbc->material.normalMap.height = image->Height;
                        }
                    }

                    if (auto& texture = material->GetMetallic().ValueMap) {
                        auto image = texture->GetTextureImage();
                        if (image) {
                            adjust_image(*image);
                            dbc->material.metallicMap.handler =
                                (TextureHandler)[m_pRenderer createTexture:*image];
                            dbc->material.metallicMap.width = image->Width;
                            dbc->material.metallicMap.height = image->Height;
                        }
                    }

                    if (auto& texture = material->GetRoughness().ValueMap) {
                        auto image = texture->GetTextureImage();
                        if (image) {
                            adjust_image(*image);
                            dbc->material.roughnessMap.handler =
                                (TextureHandler)[m_pRenderer createTexture:*image];
                            dbc->material.roughnessMap.width = image->Width;
                            dbc->material.roughnessMap.height = image->Height;
                        }
                    }

                    if (auto& texture = material->GetAO().ValueMap) {
                        auto image = texture->GetTextureImage();
                        if (image) {
                            adjust_image(*image);
                            dbc->material.aoMap.handler =
                                (TextureHandler)[m_pRenderer createTexture:*image];
                            dbc->material.aoMap.width = image->Width;
                            dbc->material.aoMap.height = image->Height;
                        }
                    }

                    material_map.emplace(material_key, dbc->material);
                }
            } else {
                dbc->material = it->second;
            }

            dbc->node = pGeometryNode;

            for (uint32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
                m_Frames[i].batchContexts.push_back(dbc);
            }

            v_property_offset += vertexPropertiesCount;
        }
    }
}

void Metal2GraphicsManager::initializeSkyBox(const Scene& scene) {
    if (scene.SkyBox) {
        std::vector<const std::shared_ptr<My::Image>> images;
        for (uint32_t i = 0; i < 18; i++) {
            auto& texture = scene.SkyBox->GetTexture(i);
            const auto& pImage = texture.GetTextureImage();
            images.push_back(pImage);
        }

        m_Frames[0].skybox = [m_pRenderer createSkyBox:images];
    }
}

void Metal2GraphicsManager::BeginFrame(Frame& frame) {
    GraphicsManager::BeginFrame(frame);

    [m_pRenderer beginFrame:frame];
}

void Metal2GraphicsManager::EndFrame(Frame& frame) {
    [m_pRenderer endFrame:frame];

    m_nFrameIndex = ((m_nFrameIndex + 1) % GfxConfiguration::kMaxInFlightFrameCount);

    GraphicsManager::EndFrame(frame);
}

void Metal2GraphicsManager::BeginPass(Frame& frame) { [m_pRenderer beginPass:frame]; }

void Metal2GraphicsManager::EndPass(Frame& frame) { [m_pRenderer endPass:frame]; }

void Metal2GraphicsManager::BeginCompute() { [m_pRenderer beginCompute]; }

void Metal2GraphicsManager::EndCompute() { [m_pRenderer endCompute]; }

void Metal2GraphicsManager::SetPipelineState(const std::shared_ptr<PipelineState>& pipelineState,
                                             const Frame& frame) {
    const std::shared_ptr<MetalPipelineState> pState =
        dynamic_pointer_cast<MetalPipelineState>(pipelineState);
    [m_pRenderer setPipelineState:*pState frameContext:frame];
}

void Metal2GraphicsManager::DrawBatch(const Frame& frame) { [m_pRenderer drawBatch:frame]; }

void Metal2GraphicsManager::GenerateCubemapArray(TextureCubeArray& texture_array) {
    [m_pRenderer generateCubemapArray:texture_array];
}

void Metal2GraphicsManager::GenerateTextureArray(Texture2DArray& texture_array) {
    [m_pRenderer generateTextureArray:texture_array];
}

void Metal2GraphicsManager::BeginShadowMap(const int32_t light_index, const TextureBase* pShadowmap,
                                           const int32_t layer_index, const Frame& frame) {
    [m_pRenderer beginShadowMap:light_index
                      shadowmap:reinterpret_cast<id<MTLTexture>>(pShadowmap->handler)
                          width:pShadowmap->width
                         height:pShadowmap->height
                    layer_index:layer_index
                          frame:frame];
}

void Metal2GraphicsManager::EndShadowMap(const TextureBase* pShadowmap, const int32_t layer_index,
                                         const Frame& frame) {
    [m_pRenderer endShadowMap:reinterpret_cast<id<MTLTexture>>(pShadowmap->handler)
                  layer_index:layer_index
                        frame:frame];
}

void Metal2GraphicsManager::SetShadowMaps(const Frame& frame) { [m_pRenderer setShadowMaps:frame]; }

void Metal2GraphicsManager::ReleaseTexture(TextureBase& texture) {
    [m_pRenderer releaseTexture:reinterpret_cast<id<MTLTexture>>(texture.handler)];
}

void Metal2GraphicsManager::DrawSkyBox(const Frame& frame) { [m_pRenderer drawSkyBox:frame]; }

void Metal2GraphicsManager::GenerateTextureForWrite(Texture2D& texture) {
    [m_pRenderer generateTextureForWrite:texture];
}

void Metal2GraphicsManager::BindTextureForWrite(Texture2D& texture, const uint32_t slot_index) {
    [m_pRenderer bindTextureForWrite:reinterpret_cast<id<MTLTexture>>(texture.handler)
                             atIndex:slot_index];
}

void Metal2GraphicsManager::Dispatch(const uint32_t width, const uint32_t height,
                                     const uint32_t depth) {
    [m_pRenderer dispatch:width height:height depth:depth];
}

void Metal2GraphicsManager::CreateTextureView(Texture2D& texture_view,
                                              const TextureArrayBase& texture_array,
                                              const uint32_t slice,
                                              const uint32_t mip) {
    [m_pRenderer createTextureView:texture_view texture_array:texture_array slice:slice mip:mip];
}