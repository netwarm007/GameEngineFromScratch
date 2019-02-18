#import <Metal/Metal.h>

#include "Metal2GraphicsManager.h"
#include "Metal2Renderer.h"

using namespace My;
using namespace std;

int Metal2GraphicsManager::Initialize()
{
    int result;

    result = GraphicsManager::Initialize();

    return result;
}

void Metal2GraphicsManager::Finalize()
{
    GraphicsManager::Finalize();
}

void Metal2GraphicsManager::Draw()
{
    GraphicsManager::Draw();
}

void Metal2GraphicsManager::Present()
{

}

void Metal2GraphicsManager::initializeGeometries(const Scene& scene)
{
    cout << "Initialize Geometries ...";
    uint32_t batch_index = 0;
    uint32_t v_property_offset = 0;
    uint32_t index_offset = 0;

    // load geometries (and materials)
    // TODO: load materials async
    for (auto& _it : scene.GeometryNodes)
    {
	    auto pGeometryNode = _it.second.lock();

        if (pGeometryNode && pGeometryNode->Visible())
        {
            auto pGeometry = scene.GetGeometry(pGeometryNode->GetSceneObjectRef());
            assert(pGeometry);
            auto pMesh = pGeometry->GetMesh().lock();
            if(!pMesh) continue;
            
            // Set the number of vertex properties.
            auto vertexPropertiesCount = pMesh->GetVertexPropertiesCount();
            
            // Set the number of vertices in the vertex array.
            // auto vertexCount = pMesh->GetVertexCount();

            for (decltype(vertexPropertiesCount) i = 0; i < vertexPropertiesCount; i++)
            {
                const SceneObjectVertexArray& v_property_array = pMesh->GetVertexPropertyArray(i);

                [m_pRenderer createVertexBuffer:v_property_array];
            }

            const SceneObjectIndexArray& index_array = pMesh->GetIndexArray(0);
            [m_pRenderer createIndexBuffer:index_array];

            MTLPrimitiveType mode;
            switch(pMesh->GetPrimitiveType())
            {
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
            switch(index_array.GetIndexType())
            {
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

            // load material textures
			if (material) {
                if (auto& texture = material->GetBaseColor().ValueMap)
                {
                    int32_t texture_id;
                    const Image& image = *texture->GetTextureImage();
                    texture_id = [m_pRenderer createTexture:image];

                    dbc->material.diffuseMap = texture_id;
                }

                if (auto& texture = material->GetNormal().ValueMap)
                {
                    int32_t texture_id;
                    const Image& image = *texture->GetTextureImage();
                    texture_id = [m_pRenderer createTexture:image];

                    dbc->material.normalMap = texture_id;
                }

                if (auto& texture = material->GetMetallic().ValueMap)
                {
                    int32_t texture_id;
                    const Image& image = *texture->GetTextureImage();
                    texture_id = [m_pRenderer createTexture:image];

                    dbc->material.metallicMap = texture_id;
                }

                if (auto& texture = material->GetRoughness().ValueMap)
                {
                    int32_t texture_id;
                    const Image& image = *texture->GetTextureImage();
                    texture_id = [m_pRenderer createTexture:image];

                    dbc->material.roughnessMap = texture_id;
                }

                if (auto& texture = material->GetAO().ValueMap)
                {
                    int32_t texture_id;
                    const Image& image = *texture->GetTextureImage();
                    texture_id = [m_pRenderer createTexture:image];

                    dbc->material.aoMap = texture_id;
                }
			}

            dbc->node = pGeometryNode;

            for (uint32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++)
            {
                m_Frames[i].batchContexts.push_back(dbc);
            }

            v_property_offset += vertexPropertiesCount;
        }
    }
}

void Metal2GraphicsManager::initializeSkyBox(const Scene& scene)
{
    if (scene.SkyBox)
    {
        std::vector<const std::shared_ptr<My::Image>> images;
        for (uint32_t i = 0; i < 18; i++)
        {
            auto& texture = scene.SkyBox->GetTexture(i);
            const auto& pImage = texture.GetTextureImage();
            images.push_back(pImage);
        }

        int32_t tex_index = [m_pRenderer createSkyBox:images];

        for (uint32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++)
        {
            m_Frames[i].frameContext.skybox = tex_index;
        }
    }
}

void Metal2GraphicsManager::initializeTerrain(const Scene& scene)
{

}

void Metal2GraphicsManager::BeginScene(const Scene& scene)
{
    GraphicsManager::BeginScene(scene);

    initializeGeometries(scene);
    initializeSkyBox(scene);
    initializeTerrain(scene);

    cout << "Done!" << endl;
}

void Metal2GraphicsManager::EndScene()
{
    GraphicsManager::EndScene();
}

void Metal2GraphicsManager::BeginFrame()
{
    [m_pRenderer beginFrame];
}

void Metal2GraphicsManager::EndFrame()
{
    [m_pRenderer endFrame];
}

void Metal2GraphicsManager::BeginPass()
{
    [m_pRenderer beginPass];
}

void Metal2GraphicsManager::EndPass()
{
    [m_pRenderer endPass];
}

void Metal2GraphicsManager::BeginCompute()
{
    [m_pRenderer beginCompute];
}

void Metal2GraphicsManager::EndCompute()
{
    [m_pRenderer endCompute];
}

void Metal2GraphicsManager::UseShaderProgram(const IShaderManager::ShaderHandler shaderProgram)
{
    [m_pRenderer useShaderProgram:shaderProgram];
}

void Metal2GraphicsManager::SetPerFrameConstants(const DrawFrameContext& context)
{
    [m_pRenderer setPerFrameConstants:context];
}

void Metal2GraphicsManager::SetPerBatchConstants(const std::vector<std::shared_ptr<DrawBatchContext>>& batches) 
{
    [m_pRenderer setPerBatchConstants:batches];
}

void Metal2GraphicsManager::SetLightInfo(const LightInfo& lightInfo)
{
    [m_pRenderer setLightInfo:lightInfo];
}

void Metal2GraphicsManager::DrawBatch(const std::vector<std::shared_ptr<DrawBatchContext>>& batches)
{
    [m_pRenderer drawBatch:batches];
}

int32_t Metal2GraphicsManager::GenerateCubeShadowMapArray(const uint32_t width, const uint32_t height, const uint32_t count) 
{
    return [m_pRenderer generateCubeShadowMapArray:width height:height count:count];
}

int32_t Metal2GraphicsManager::GenerateShadowMapArray(const uint32_t width, const uint32_t height, const uint32_t count) 
{
    return [m_pRenderer generateShadowMapArray:width height:height count:count];
}

void Metal2GraphicsManager::BeginShadowMap(const Light& light, const int32_t shadowmap, const uint32_t width, const uint32_t height, const int32_t layer_index) 
{
    [m_pRenderer beginShadowMap:light shadowmap:shadowmap width:width height:height layer_index:layer_index];
}

void Metal2GraphicsManager::EndShadowMap(const int32_t shadowmap, const int32_t layer_index) 
{
    [m_pRenderer endShadowMap:shadowmap layer_index:layer_index];
}

void Metal2GraphicsManager::SetShadowMaps(const Frame& frame) 
{
    [m_pRenderer setShadowMaps:frame];
}

void Metal2GraphicsManager::DestroyShadowMap(int32_t& shadowmap) 
{
    [m_pRenderer destroyShadowMap:shadowmap];
}

void Metal2GraphicsManager::SetSkyBox(const DrawFrameContext& context)
{
    [m_pRenderer setSkyBox:context];
}

void Metal2GraphicsManager::DrawSkyBox()
{
    [m_pRenderer drawSkyBox];
}

int32_t Metal2GraphicsManager::GenerateAndBindTextureForWrite(const char* id, const uint32_t slot_index,
                                        const uint32_t width, const uint32_t height)
{
    return [m_pRenderer generateAndBindTextureForWrite:width height:height atIndex:slot_index];
}

void Metal2GraphicsManager::Dispatch(const uint32_t width, const uint32_t height, const uint32_t depth)
{
    [m_pRenderer dispatch:width height:height depth:depth];
}
