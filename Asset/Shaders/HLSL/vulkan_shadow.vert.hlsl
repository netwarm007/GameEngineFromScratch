cbuffer PerBatchConstants : register(b11)
{
    row_major float4x4 modelMatrix;
};

struct Light {
    row_major float4x4 lightViewMatrix;
    row_major float4x4 lightProjectionMatrix;
    float lightIntensity;
    int lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    float2 lightSize;
    int4 lightGuid;
    float4 lightPosition;
    float4 lightColor;
    float4 lightDirection;
    float4 lightDistAttenCurveParams[2];
    float4 lightAngleAttenCurveParams[2];
};

StructuredBuffer<Light> lights : register(t12);

cbuffer ShadowMapConstants : register(b13)
{
    int light_index;
    float shadowmap_layer_index;
    float near_plane;
    float far_plane;
    row_major float4x4 shadowMatrices[6];
};

struct ShadowVertexOutput
{
    float4 position : SV_Position;
    float3 worldPosition : TEXCOORD0;
};

ShadowVertexOutput vulkan_shadow_vert_main(float3 position : POSITION)
{
    float4 worldPosition = mul(float4(position, 1.0f), modelMatrix);
    ShadowVertexOutput output;
    output.worldPosition = worldPosition.xyz;
    if (lights[light_index].lightType == 0) {
        output.position =
            mul(worldPosition, shadowMatrices[int(shadowmap_layer_index)]);
    } else {
        float4 lightPosition =
            mul(worldPosition, lights[light_index].lightViewMatrix);
        output.position =
            mul(lightPosition, lights[light_index].lightProjectionMatrix);
    }
    return output;
}
