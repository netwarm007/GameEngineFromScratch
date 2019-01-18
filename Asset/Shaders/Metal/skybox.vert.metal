#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct a2v_pos_only
{
    float3 inputPosition;
};

struct cube_vert_output
{
    float4 pos;
    float3 uvw;
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

struct skybox_vert_main_out
{
    float3 _entryPointOutput_uvw [[user(locn0)]];
    float4 gl_Position [[position]];
};

struct skybox_vert_main_in
{
    float3 a_inputPosition [[attribute(0)]];
};

cube_vert_output _skybox_vert_main(thread const a2v_pos_only& a, constant PerFrameConstants& v_30)
{
    cube_vert_output o;
    o.uvw = a.inputPosition;
    float4x4 _matrix = v_30.viewMatrix;
    _matrix[3].x = 0.0;
    _matrix[3].y = 0.0;
    _matrix[3].z = 0.0;
    float4 pos = v_30.projectionMatrix * (_matrix * float4(a.inputPosition, 1.0));
    o.pos = pos.xyww;
    return o;
}

vertex skybox_vert_main_out skybox_vert_main(skybox_vert_main_in in [[stage_in]], constant PerFrameConstants& v_30 [[buffer(10)]])
{
    skybox_vert_main_out out = {};
    a2v_pos_only a;
    a.inputPosition = in.a_inputPosition;
    a2v_pos_only param = a;
    cube_vert_output flattenTemp = _skybox_vert_main(param, v_30);
    out.gl_Position = flattenTemp.pos;
    out._entryPointOutput_uvw = flattenTemp.uvw;
    return out;
}

