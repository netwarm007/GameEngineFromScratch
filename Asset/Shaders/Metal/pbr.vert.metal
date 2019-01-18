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

struct pbr_vert_output
{
    float4 pos;
    float4 normal;
    float4 normal_world;
    float4 v;
    float4 v_world;
    float3 v_tangent;
    float3 camPos_tangent;
    float2 uv;
    float3x3 TBN;
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

struct pbr_vert_main_out
{
    float4 _entryPointOutput_normal [[user(locn0)]];
    float4 _entryPointOutput_normal_world [[user(locn1)]];
    float4 _entryPointOutput_v [[user(locn2)]];
    float4 _entryPointOutput_v_world [[user(locn3)]];
    float3 _entryPointOutput_v_tangent [[user(locn4)]];
    float3 _entryPointOutput_camPos_tangent [[user(locn5)]];
    float2 _entryPointOutput_uv [[user(locn6)]];
    float3 _entryPointOutput_TBN_0 [[user(locn7)]];
    float3 _entryPointOutput_TBN_1 [[user(locn8)]];
    float3 _entryPointOutput_TBN_2 [[user(locn9)]];
    float4 gl_Position [[position]];
};

struct pbr_vert_main_in
{
    float3 a_inputPosition [[attribute(0)]];
    float3 a_inputNormal [[attribute(1)]];
    float2 a_inputUV [[attribute(2)]];
    float3 a_inputTangent [[attribute(3)]];
};

pbr_vert_output _pbr_vert_main(thread const a2v& a, constant PerBatchConstants& v_25, constant PerFrameConstants& v_44)
{
    pbr_vert_output o;
    o.v_world = v_25.modelMatrix * float4(a.inputPosition, 1.0);
    o.v = v_44.viewMatrix * o.v_world;
    o.pos = v_44.projectionMatrix * o.v;
    o.normal_world = normalize(v_25.modelMatrix * float4(a.inputNormal, 0.0));
    o.normal = normalize(v_44.viewMatrix * o.normal_world);
    float3 tangent = normalize((v_25.modelMatrix * float4(a.inputTangent, 0.0)).xyz);
    tangent = normalize(tangent - (o.normal_world.xyz * dot(tangent, o.normal_world.xyz)));
    float3 bitangent = cross(o.normal_world.xyz, tangent);
    o.TBN = float3x3(float3(tangent), float3(bitangent), float3(o.normal_world.xyz));
    float3x3 TBN_trans = transpose(o.TBN);
    o.v_tangent = TBN_trans * o.v_world.xyz;
    o.camPos_tangent = TBN_trans * v_44.camPos.xyz;
    o.uv.x = a.inputUV.x;
    o.uv.y = 1.0 - a.inputUV.y;
    return o;
}

vertex pbr_vert_main_out pbr_vert_main(pbr_vert_main_in in [[stage_in]], constant PerFrameConstants& v_44 [[buffer(10)]], constant PerBatchConstants& v_25 [[buffer(11)]])
{
    pbr_vert_main_out out = {};
    float3x3 _entryPointOutput_TBN = {};
    a2v a;
    a.inputPosition = in.a_inputPosition;
    a.inputNormal = in.a_inputNormal;
    a.inputUV = in.a_inputUV;
    a.inputTangent = in.a_inputTangent;
    a2v param = a;
    pbr_vert_output flattenTemp = _pbr_vert_main(param, v_25, v_44);
    out.gl_Position = flattenTemp.pos;
    out._entryPointOutput_normal = flattenTemp.normal;
    out._entryPointOutput_normal_world = flattenTemp.normal_world;
    out._entryPointOutput_v = flattenTemp.v;
    out._entryPointOutput_v_world = flattenTemp.v_world;
    out._entryPointOutput_v_tangent = flattenTemp.v_tangent;
    out._entryPointOutput_camPos_tangent = flattenTemp.camPos_tangent;
    out._entryPointOutput_uv = flattenTemp.uv;
    _entryPointOutput_TBN = flattenTemp.TBN;
    out._entryPointOutput_TBN_0 = _entryPointOutput_TBN[0];
    out._entryPointOutput_TBN_1 = _entryPointOutput_TBN[1];
    out._entryPointOutput_TBN_2 = _entryPointOutput_TBN[2];
    return out;
}

