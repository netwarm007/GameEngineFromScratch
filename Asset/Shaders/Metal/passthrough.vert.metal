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

struct simple_vert_output
{
    float4 pos;
    float2 uv;
};

struct PerFrameConstants
{
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4 camPos;
    int numLights;
};

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

struct DebugConstants
{
    int layer_index;
    float mip_level;
    float line_width;
    float padding0;
    float4 front_color;
    float4 back_color;
};

struct ShadowMapConstants
{
    int shadowmap_layer_index;
    float far_plane;
    float padding[2];
    float4 lightPos;
    float4x4 lightVP;
    float4x4 shadowMatrices[6];
};

struct passthrough_vert_main_out
{
    float2 _entryPointOutput_uv [[user(locn0)]];
    float4 gl_Position [[position]];
};

struct passthrough_vert_main_in
{
    float3 a_inputPosition [[attribute(0)]];
    float3 a_inputNormal [[attribute(1)]];
    float2 a_inputUV [[attribute(2)]];
    float3 a_inputTangent [[attribute(3)]];
    float3 a_inputBiTangent [[attribute(4)]];
};

simple_vert_output _passthrough_vert_main(thread const a2v& a)
{
    simple_vert_output o;
    o.pos = float4(a.inputPosition, 1.0);
    o.uv = a.inputUV;
    return o;
}

vertex passthrough_vert_main_out passthrough_vert_main(passthrough_vert_main_in in [[stage_in]])
{
    passthrough_vert_main_out out = {};
    a2v a;
    a.inputPosition = in.a_inputPosition;
    a.inputNormal = in.a_inputNormal;
    a.inputUV = in.a_inputUV;
    a.inputTangent = in.a_inputTangent;
    a.inputBiTangent = in.a_inputBiTangent;
    a2v param = a;
    simple_vert_output flattenTemp = _passthrough_vert_main(param);
    out.gl_Position = flattenTemp.pos;
    out._entryPointOutput_uv = flattenTemp.uv;
    return out;
}

