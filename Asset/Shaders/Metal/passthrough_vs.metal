#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

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

struct passthrough_vs_main_out
{
    float2 UV [[user(locn0)]];
    float4 gl_Position [[position]];
};

struct passthrough_vs_main_in
{
    float3 inputPosition [[attribute(0)]];
    float2 inputUV [[attribute(1)]];
};

vertex passthrough_vs_main_out passthrough_vs_main(passthrough_vs_main_in in [[stage_in]])
{
    passthrough_vs_main_out out = {};
    out.gl_Position = float4(in.inputPosition, 1.0);
    out.UV = in.inputUV;
    return out;
}

