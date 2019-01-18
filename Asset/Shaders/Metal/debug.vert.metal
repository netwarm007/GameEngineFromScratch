#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct a2v_pos_only
{
    float3 inputPosition;
};

struct pos_only_vert_output
{
    float4 pos;
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

struct debug_vert_main_out
{
    float4 gl_Position [[position]];
};

struct debug_vert_main_in
{
    float3 a_inputPosition [[attribute(0)]];
};

pos_only_vert_output _debug_vert_main(thread const a2v_pos_only& a, constant PerFrameConstants& v_31)
{
    float4 v = float4(a.inputPosition, 1.0);
    v = v_31.viewMatrix * v;
    pos_only_vert_output o;
    o.pos = v_31.projectionMatrix * v;
    return o;
}

vertex debug_vert_main_out debug_vert_main(debug_vert_main_in in [[stage_in]], constant PerFrameConstants& v_31 [[buffer(10)]])
{
    debug_vert_main_out out = {};
    a2v_pos_only a;
    a.inputPosition = in.a_inputPosition;
    a2v_pos_only param = a;
    out.gl_Position = _debug_vert_main(param, v_31).pos;
    return out;
}

