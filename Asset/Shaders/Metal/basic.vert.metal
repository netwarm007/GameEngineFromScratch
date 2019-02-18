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
};

struct basic_vert_output
{
    float4 pos;
    float4 normal;
    float4 normal_world;
    float4 v;
    float4 v_world;
    float2 uv;
};

struct PerBatchConstants
{
    float4x4 modelMatrix;
};

struct PerFrameConstants
{
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
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

struct DebugConstants
{
    float layer_index;
    float mip_level;
    float line_width;
    float padding0;
    float4 front_color;
    float4 back_color;
};

struct ShadowMapConstants
{
    float4x4 shadowMatrices[6];
    float shadowmap_layer_index;
    float far_plane;
    float padding[2];
    float4 lightPos;
};

struct basic_vert_main_out
{
    float4 _entryPointOutput_normal [[user(locn0)]];
    float4 _entryPointOutput_normal_world [[user(locn1)]];
    float4 _entryPointOutput_v [[user(locn2)]];
    float4 _entryPointOutput_v_world [[user(locn3)]];
    float2 _entryPointOutput_uv [[user(locn4)]];
    float4 gl_Position [[position]];
};

struct basic_vert_main_in
{
    float3 a_inputPosition [[attribute(0)]];
    float3 a_inputNormal [[attribute(1)]];
    float2 a_inputUV [[attribute(2)]];
    float3 a_inputTangent [[attribute(3)]];
};

basic_vert_output _basic_vert_main(thread const a2v& a, constant PerBatchConstants& v_24, constant PerFrameConstants& v_43)
{
    basic_vert_output o;
    o.v_world = v_24.modelMatrix * float4(a.inputPosition, 1.0);
    o.v = v_43.viewMatrix * o.v_world;
    o.pos = v_43.projectionMatrix * o.v;
    o.normal_world = normalize(v_24.modelMatrix * float4(a.inputNormal, 0.0));
    o.normal = normalize(v_43.viewMatrix * o.normal_world);
    o.uv.x = a.inputUV.x;
    o.uv.y = 1.0 - a.inputUV.y;
    return o;
}

vertex basic_vert_main_out basic_vert_main(basic_vert_main_in in [[stage_in]], constant PerFrameConstants& v_43 [[buffer(10)]], constant PerBatchConstants& v_24 [[buffer(11)]])
{
    basic_vert_main_out out = {};
    a2v a;
    a.inputPosition = in.a_inputPosition;
    a.inputNormal = in.a_inputNormal;
    a.inputUV = in.a_inputUV;
    a.inputTangent = in.a_inputTangent;
    a2v param = a;
    basic_vert_output flattenTemp = _basic_vert_main(param, v_24, v_43);
    out.gl_Position = flattenTemp.pos;
    out._entryPointOutput_normal = flattenTemp.normal;
    out._entryPointOutput_normal_world = flattenTemp.normal_world;
    out._entryPointOutput_v = flattenTemp.v;
    out._entryPointOutput_v_world = flattenTemp.v_world;
    out._entryPointOutput_uv = flattenTemp.uv;
    return out;
}

