#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct debugPushConstants
{
    float layer_index;
};

struct Light
{
    float lightIntensity;
    int lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    float2 lightSize;
    int4 lightGUID;
    float4 lightPosition;
    float4 lightColor;
    float4 lightDirection;
    float4 lightDistAttenCurveParams[2];
    float4 lightAngleAttenCurveParams[2];
    float4x4 lightVP;
    float4 padding[2];
};

struct PerFrameConstants
{
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4 camPos;
    int numLights;
    Light allLights[100];
};

struct PerBatchConstants
{
    float4x4 modelMatrix;
};

struct depthtexturearray_frag_main_out
{
    float3 color [[color(0)]];
};

struct depthtexturearray_frag_main_in
{
    float2 UV [[user(locn0)]];
};

fragment depthtexturearray_frag_main_out depthtexturearray_frag_main(depthtexturearray_frag_main_in in [[stage_in]], constant debugPushConstants& u_pushConstants [[buffer(0)]], texture2d_array<float> depthSampler [[texture(0)]], sampler depthSamplerSmplr [[sampler(0)]])
{
    depthtexturearray_frag_main_out out = {};
    out.color = depthSampler.sample(depthSamplerSmplr, float3(in.UV, u_pushConstants.layer_index).xy, uint(round(float3(in.UV, u_pushConstants.layer_index).z))).xxx;
    return out;
}

