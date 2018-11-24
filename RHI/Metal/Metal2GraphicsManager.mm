#import <Metal/Metal.h>

#include "Metal2GraphicsManager.h"

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

bool Metal2GraphicsManager::CheckCapability(RHICapability cap)
{
    return true;
}

void Metal2GraphicsManager::BeginScene(const Scene& scene)
{
    cout << "Creating Draw Batch Contexts ...";
    uint32_t batch_index = 0;
    for (auto& _it : scene.GeometryNodes)
    {
	    auto pGeometryNode = _it.second.lock();

        uint32_t v_property_offset = 0;
        uint32_t index_offset = 0;
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

			auto material_index = index_array.GetMaterialIndex();
			auto material_key = pGeometryNode->GetMaterialRef(material_index);
			auto material = scene.GetMaterial(material_key);

            auto dbc = make_shared<MtlDrawBatchContext>();
            dbc->batchIndex = batch_index++;
            dbc->index_offset = index_offset++;
			dbc->index_count = (uint32_t)index_array.GetIndexCount();
            dbc->property_offset = v_property_offset;
            dbc->property_count = vertexPropertiesCount;

			if (material) {
                if (auto& texture = material->GetBaseColor().ValueMap)
                {
                    int32_t texture_id;
                    const auto& it = m_TextureIndex.find(texture->GetName());
                    if (it == m_TextureIndex.cend()) {
                        const Image& image = *texture->GetTextureImage();
                        texture_id = [m_pRenderer createTexture:image];
                        m_TextureIndex[texture->GetName()] = texture_id;
                    }
                    else
                    {
                        texture_id = it->second;
                    }
                    dbc->material.diffuseMap = texture_id;
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
    cout << "Done!" << endl;
}

void Metal2GraphicsManager::EndScene()
{

}

void Metal2GraphicsManager::BeginFrame()
{
    [m_pRenderer beginFrame];
}

void Metal2GraphicsManager::EndFrame()
{
    [m_pRenderer endFrame];
}

void Metal2GraphicsManager::UseShaderProgram(const intptr_t shaderProgram)
{

}

void Metal2GraphicsManager::SetPerFrameConstants(const DrawFrameContext& context)
{
    [m_pRenderer setPerFrameConstants:context];
}

void Metal2GraphicsManager::SetPerBatchConstants(const DrawBatchContext& context)
{
    [m_pRenderer setPerBatchConstants:context];
}

void Metal2GraphicsManager::DrawBatch(const DrawBatchContext& context)
{
    const MtlDrawBatchContext& dbc = dynamic_cast<const MtlDrawBatchContext&>(context);
    [m_pRenderer drawBatch:dbc];
}
