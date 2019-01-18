#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct cube_vert_output
{
    float4 pos;
    float3 uvw;
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

struct ShadowMapConstants
{
    float4x4 shadowMatrices[6];
    float shadowmap_layer_index;
    float far_plane;
    float padding[2];
    float4 lightPos;
};

struct cubemap_frag_main_out
{
    float4 _entryPointOutput [[color(0)]];
};

struct cubemap_frag_main_in
{
    float3 _entryPointOutput_uvw [[user(locn0)]];
};

float4 _cubemap_frag_main(thread const cube_vert_output& _entryPointOutput, thread texturecube<float> cubemap, thread sampler samp0, constant DebugConstants& v_32)
{
    return cubemap.sample(samp0, _entryPointOutput.uvw, level(v_32.mip_level));
}

fragment cubemap_frag_main_out cubemap_frag_main(cubemap_frag_main_in in [[stage_in]], constant DebugConstants& v_32 [[buffer(13)]], texturecube<float> cubemap [[texture(0)]], sampler samp0 [[sampler(0)]], float4 gl_FragCoord [[position]])
{
    cubemap_frag_main_out out = {};
    cube_vert_output _entryPointOutput;
    _entryPointOutput.pos = gl_FragCoord;
    _entryPointOutput.uvw = in._entryPointOutput_uvw;
    cube_vert_output param = _entryPointOutput;
    out._entryPointOutput = _cubemap_frag_main(param, cubemap, samp0, v_32);
    return out;
}

