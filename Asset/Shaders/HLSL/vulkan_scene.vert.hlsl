cbuffer PerFrameConstants : register(b10)
{
    row_major float4x4 viewMatrix;
    row_major float4x4 projectionMatrix;
    float4 camPos;
    int numLights;
    int clip_space_type;
};

cbuffer PerBatchConstants : register(b11)
{
    row_major float4x4 modelMatrix;
    float4 baseColor;
    float4 materialParameters;
};

struct VulkanSceneVertex {
    float3 inputPosition : POSITION;
    float3 inputNormal : NORMAL;
    float2 inputTexCoord : TEXCOORD0;
    float3 inputTangent : TANGENT;
};

struct VulkanSceneVertexOutput {
    float4 position : SV_POSITION;
    float2 texCoords : TEXCOORD0;
    float3 worldPosition : TEXCOORD1;
    float3 worldNormal : TEXCOORD2;
    float3 worldTangent : TEXCOORD3;
};

VulkanSceneVertexOutput vulkan_scene_vert_main(VulkanSceneVertex input)
{
    VulkanSceneVertexOutput output;
    float4 worldPosition = mul(float4(input.inputPosition, 1.0f), modelMatrix);
    output.position = mul(mul(worldPosition, viewMatrix), projectionMatrix);
    output.texCoords =
        float2(input.inputTexCoord.x, 1.0f - input.inputTexCoord.y);
    output.worldPosition = worldPosition.xyz;
    output.worldNormal =
        normalize(mul(float4(input.inputNormal, 0.0f), modelMatrix).xyz);
    float3 tangent =
        mul(float4(input.inputTangent, 0.0f), modelMatrix).xyz;
    output.worldTangent =
        normalize(tangent - output.worldNormal *
                              dot(output.worldNormal, tangent));
    return output;
}
