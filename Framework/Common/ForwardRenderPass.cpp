#include "ForwardRenderPass.hpp"
#include "GraphicsManager.hpp"
#include "IShaderManager.hpp"
#include "IPhysicsManager.hpp"

using namespace My;
using namespace std;

void ForwardRenderPass::Draw(const Frame& frame)
{
    auto shaderProgram = g_pShaderManager->GetDefaultShaderProgram();

    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    g_pGraphicsManager->UseShaderProgram(shaderProgram);

    g_pGraphicsManager->SetPerFrameConstants(frame.frameContext);

    for (auto dbc : frame.batchContexts)
    {
        if (void* rigidBody = dbc->node->RigidBody()) {
            Matrix4X4f trans;

            // the geometry has rigid body bounded, we blend the simlation result here.
            Matrix4X4f simulated_result = g_pPhysicsManager->GetRigidBodyTransform(rigidBody);

            BuildIdentityMatrix(trans);

            // apply the rotation part of the simlation result
            memcpy(trans[0], simulated_result[0], sizeof(float) * 3);
            memcpy(trans[1], simulated_result[1], sizeof(float) * 3);
            memcpy(trans[2], simulated_result[2], sizeof(float) * 3);

            // replace the translation part of the matrix with simlation result directly
            memcpy(trans[3], simulated_result[3], sizeof(float) * 3);

            dbc->trans = trans;
        } else {
            dbc->trans = *dbc->node->GetCalculatedTransform();
        }

        g_pGraphicsManager->DrawBatch(*dbc);
    }
}
