#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct debugPushConstants
{
    float level;
};

struct Light
{
    int lightType;
    float lightIntensity;
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

struct main0_out
{
    float3 color [[color(0)]];
};

struct main0_in
{
    float3 UVW [[user(locn0)]];
};

fragment main0_out main0(main0_in in [[stage_in]], constant debugPushConstants& u_pushConstants [[buffer(0)]], texturecube<float> depthSampler [[texture(0)]], sampler depthSamplerSmplr [[sampler(0)]])
{
    main0_out out = {};
    out.color = depthSampler.sample(depthSamplerSmplr, in.UVW, level(u_pushConstants.level)).xyz;
    return out;
}

