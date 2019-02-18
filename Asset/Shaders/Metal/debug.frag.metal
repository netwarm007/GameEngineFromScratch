#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

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

struct debug_frag_main_out
{
    float4 _entryPointOutput [[color(0)]];
};

float4 _debug_frag_main(thread const pos_only_vert_output& _entryPointOutput)
{
    return float4(1.0);
}

fragment debug_frag_main_out debug_frag_main(float4 gl_FragCoord [[position]])
{
    debug_frag_main_out out = {};
    pos_only_vert_output _entryPointOutput;
    _entryPointOutput.pos = gl_FragCoord;
    pos_only_vert_output param = _entryPointOutput;
    out._entryPointOutput = _debug_frag_main(param);
    return out;
}

