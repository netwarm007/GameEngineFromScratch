#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct PerBatchConstants
{
    float4x4 modelMatrix;
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

struct basic_vs_main_out
{
    float4 normal [[user(locn0)]];
    float4 normal_world [[user(locn1)]];
    float4 v [[user(locn2)]];
    float4 v_world [[user(locn3)]];
    float2 uv [[user(locn4)]];
    float4 gl_Position [[position]];
};

struct basic_vs_main_in
{
    float3 inputPosition [[attribute(0)]];
    float3 inputNormal [[attribute(1)]];
    float2 inputUV [[attribute(2)]];
};

vertex basic_vs_main_out basic_vs_main(basic_vs_main_in in [[stage_in]], constant PerFrameConstants& _42 [[buffer(0)]], constant PerBatchConstants& _13 [[buffer(1)]])
{
    basic_vs_main_out out = {};
    out.v_world = _13.modelMatrix * float4(in.inputPosition, 1.0);
    out.v = _42.viewMatrix * out.v_world;
    out.gl_Position = _42.projectionMatrix * out.v;
    out.normal_world = _13.modelMatrix * float4(in.inputNormal, 0.0);
    out.normal = _42.viewMatrix * out.normal_world;
    out.uv.x = in.inputUV.x;
    out.uv.y = 1.0 - in.inputUV.y;
    return out;
}

