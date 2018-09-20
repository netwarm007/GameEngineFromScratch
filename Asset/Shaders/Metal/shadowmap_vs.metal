#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct constants_t
{
    float4x4 depthVP;
};

struct PerBatchConstants
{
    float4x4 modelMatrix;
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

struct main0_out
{
    float4 gl_Position [[position]];
};

struct main0_in
{
    float3 inputPosition [[attribute(0)]];
};

vertex main0_out main0(main0_in in [[stage_in]], constant constants_t& u_pushConstants [[buffer(0)]], constant PerBatchConstants& _25 [[buffer(1)]])
{
    main0_out out = {};
    out.gl_Position = (u_pushConstants.depthVP * _25.modelMatrix) * float4(in.inputPosition, 1.0);
    return out;
}

