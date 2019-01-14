#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct a2v
{
    float3 inputPosition;
    float3 inputNormal;
    float2 inputUV;
    float3 inputTangent;
    float3 inputBiTangent;
};

struct pos_only_vert_output
{
    float4 pos;
};

struct PerBatchConstants
{
    float4x4 modelMatrix;
};

struct PerFrameConstants
{
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 arbitraryMatrix;
    float4 camPos;
    int numLights;
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
    int4 lightGuid;
    float4 lightPosition;
    float4 lightColor;
    float4 lightDirection;
    float4 lightDistAttenCurveParams[2];
    float4 lightAngleAttenCurveParams[2];
    float4x4 lightVP;
    float4 padding[2];
};

struct LightInfo
{
    Light lights[100];
};

struct shadowmap_vert_main_out
{
    float4 gl_Position [[position]];
};

struct shadowmap_vert_main_in
{
    float3 a_inputPosition [[attribute(0)]];
    float3 a_inputNormal [[attribute(1)]];
    float2 a_inputUV [[attribute(2)]];
    float3 a_inputTangent [[attribute(3)]];
    float3 a_inputBiTangent [[attribute(4)]];
};

pos_only_vert_output _shadowmap_vert_main(thread const a2v& a, constant PerBatchConstants& v_32, constant PerFrameConstants& v_42)
{
    float4 v = float4(a.inputPosition, 1.0);
    v = v_32.modelMatrix * v;
    pos_only_vert_output o;
    o.pos = v_42.arbitraryMatrix * v;
    return o;
}

vertex shadowmap_vert_main_out shadowmap_vert_main(shadowmap_vert_main_in in [[stage_in]], constant PerFrameConstants& v_42 [[buffer(10)]], constant PerBatchConstants& v_32 [[buffer(11)]])
{
    shadowmap_vert_main_out out = {};
    a2v a;
    a.inputPosition = in.a_inputPosition;
    a.inputNormal = in.a_inputNormal;
    a.inputUV = in.a_inputUV;
    a.inputTangent = in.a_inputTangent;
    a.inputBiTangent = in.a_inputBiTangent;
    a2v param = a;
    out.gl_Position = _shadowmap_vert_main(param, v_32, v_42).pos;
    return out;
}

