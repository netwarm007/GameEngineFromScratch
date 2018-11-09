#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct debugPushConstants
{
    float3 FrontColor;
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

struct debug_ps_main_out
{
    float4 outputColor [[color(0)]];
};

fragment debug_ps_main_out debug_ps_main(constant debugPushConstants& u_pushConstants [[buffer(0)]])
{
    debug_ps_main_out out = {};
    out.outputColor = float4(u_pushConstants.FrontColor, 1.0);
    return out;
}

