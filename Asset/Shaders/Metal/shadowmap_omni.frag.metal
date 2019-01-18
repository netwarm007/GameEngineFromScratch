#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct pos_only_vert_output
{
    float4 pos;
};

struct ShadowMapConstants
{
    float4x4 shadowMatrices[6];
    float4 lightPos;
    float shadowmap_layer_index;
    float far_plane;
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

struct shadowmap_omni_frag_main_out
{
    float gl_FragDepth [[depth(any)]];
};

float _shadowmap_omni_frag_main(thread const pos_only_vert_output& _entryPointOutput, constant ShadowMapConstants& v_29)
{
    float lightDistance = length(_entryPointOutput.pos.xyz - float3(v_29.lightPos.xyz));
    lightDistance /= v_29.far_plane;
    return lightDistance;
}

fragment shadowmap_omni_frag_main_out shadowmap_omni_frag_main(constant ShadowMapConstants& v_29 [[buffer(14)]], float4 gl_FragCoord [[position]])
{
    shadowmap_omni_frag_main_out out = {};
    pos_only_vert_output _entryPointOutput;
    _entryPointOutput.pos = gl_FragCoord;
    pos_only_vert_output param = _entryPointOutput;
    out.gl_FragDepth = _shadowmap_omni_frag_main(param, v_29);
    return out;
}

