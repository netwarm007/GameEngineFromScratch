#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct ps_constant_t
{
    packed_float3 lightPos;
    float far_plane;
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
    float gl_FragDepth [[depth(any)]];
};

struct main0_in
{
    float4 FragPos [[user(locn0)]];
};

fragment main0_out main0(main0_in in [[stage_in]], constant ps_constant_t& u_lightParams [[buffer(0)]])
{
    main0_out out = {};
    float lightDistance = length(in.FragPos.xyz - float3(u_lightParams.lightPos));
    lightDistance /= u_lightParams.far_plane;
    out.gl_FragDepth = lightDistance;
    return out;
}

