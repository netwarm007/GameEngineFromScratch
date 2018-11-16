#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct vert_output
{
    float4 position;
    float4 normal;
    float4 normal_world;
    float4 v;
    float4 v_world;
    float2 uv;
    float3x3 TBN;
    float3 v_tangent;
    float3 camPos_tangent;
};

struct Light
{
    float lightIntensity;
    uint lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    uint lightAngleAttenCurveType;
    uint lightDistAttenCurveType;
    float2 lightSize;
    uint4 lightGuid;
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
    uint numLights;
    float padding[3];
    Light lights[100];
};

struct PerBatchConstants
{
    float4x4 modelMatrix;
};

struct debug_frag_main_out
{
    float4 _entryPointOutput [[color(0)]];
};

struct debug_frag_main_in
{
    float4 input_normal [[user(locn0)]];
    float4 input_normal_world [[user(locn1)]];
    float4 input_v [[user(locn2)]];
    float4 input_v_world [[user(locn3)]];
    float2 input_uv [[user(locn4)]];
    float3 input_TBN_0 [[user(locn5)]];
    float3 input_TBN_1 [[user(locn6)]];
    float3 input_TBN_2 [[user(locn7)]];
    float3 input_v_tangent [[user(locn8)]];
    float3 input_camPos_tangent [[user(locn9)]];
};

float4 _debug_frag_main(thread const vert_output& _input)
{
    return float4(1.0);
}

fragment debug_frag_main_out debug_frag_main(debug_frag_main_in in [[stage_in]], float4 gl_FragCoord [[position]])
{
    debug_frag_main_out out = {};
    float3x3 input_TBN = {};
    input_TBN[0] = in.input_TBN_0;
    input_TBN[1] = in.input_TBN_1;
    input_TBN[2] = in.input_TBN_2;
    vert_output _input;
    _input.position = gl_FragCoord;
    _input.normal = in.input_normal;
    _input.normal_world = in.input_normal_world;
    _input.v = in.input_v;
    _input.v_world = in.input_v_world;
    _input.uv = in.input_uv;
    _input.TBN = input_TBN;
    _input.v_tangent = in.input_v_tangent;
    _input.camPos_tangent = in.input_camPos_tangent;
    vert_output param = _input;
    out._entryPointOutput = _debug_frag_main(param);
    return out;
}

