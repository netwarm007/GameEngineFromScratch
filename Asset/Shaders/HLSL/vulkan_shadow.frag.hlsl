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

float vulkan_shadow_frag_main(float4 position : SV_Position,
                              float3 worldPosition : TEXCOORD0) : SV_Depth
{
    if (lights[light_index].lightType == 0) {
        return saturate(length(worldPosition -
                               lights[light_index].lightPosition.xyz) /
                        far_plane);
    }
    return position.z;
}
