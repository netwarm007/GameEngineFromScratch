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
    uint numLights;
};

struct PerBatchConstants
{
    float4x4 modelMatrix;
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

struct LightInfo
{
    Light lights[100];
};

struct debug_frag_main_out
{
    float4 _entryPointOutput [[color(0)]];
};

float4 _debug_frag_main(thread const pos_only_vert_output& _input)
{
    return float4(1.0);
}

fragment debug_frag_main_out debug_frag_main(float4 gl_FragCoord [[position]])
{
    debug_frag_main_out out = {};
    pos_only_vert_output _input;
    _input.pos = gl_FragCoord;
    pos_only_vert_output param = _input;
    out._entryPointOutput = _debug_frag_main(param);
    return out;
}

