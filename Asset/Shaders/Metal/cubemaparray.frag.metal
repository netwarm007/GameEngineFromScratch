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
    int shadowmap_layer_index;
    float far_plane;
    float padding[2];
    float4x4 lightVP;
    float4x4 shadowMatrices[6];
};

struct cubemaparray_frag_main_out
{
    float4 _entryPointOutput [[color(0)]];
};

struct cubemaparray_frag_main_in
{
    float3 _entryPointOutput_uvw [[user(locn0)]];
};

float3 convert_xyz_to_cube_uv(thread const float3& d)
{
    float3 d_abs = abs(d);
    bool3 isPositive;
    isPositive.x = int(d.x > 0.0) != 0u;
    isPositive.y = int(d.y > 0.0) != 0u;
    isPositive.z = int(d.z > 0.0) != 0u;
    float maxAxis;
    float uc;
    float vc;
    int index;
    if ((isPositive.x && (d_abs.x >= d_abs.y)) && (d_abs.x >= d_abs.z))
    {
        maxAxis = d_abs.x;
        uc = -d.z;
        vc = d.y;
        index = 0;
    }
    if (((!isPositive.x) && (d_abs.x >= d_abs.y)) && (d_abs.x >= d_abs.z))
    {
        maxAxis = d_abs.x;
        uc = d.z;
        vc = d.y;
        index = 1;
    }
    if ((isPositive.y && (d_abs.y >= d_abs.x)) && (d_abs.y >= d_abs.z))
    {
        maxAxis = d_abs.y;
        uc = d.x;
        vc = -d.z;
        index = 3;
    }
    if (((!isPositive.y) && (d_abs.y >= d_abs.x)) && (d_abs.y >= d_abs.z))
    {
        maxAxis = d_abs.y;
        uc = d.x;
        vc = d.z;
        index = 2;
    }
    if ((isPositive.z && (d_abs.z >= d_abs.x)) && (d_abs.z >= d_abs.y))
    {
        maxAxis = d_abs.z;
        uc = d.x;
        vc = d.y;
        index = 4;
    }
    if (((!isPositive.z) && (d_abs.z >= d_abs.x)) && (d_abs.z >= d_abs.y))
    {
        maxAxis = d_abs.z;
        uc = -d.x;
        vc = d.y;
        index = 5;
    }
    float3 o;
    o.x = 0.5 * ((uc / maxAxis) + 1.0);
    o.y = 0.5 * ((vc / maxAxis) + 1.0);
    o.z = float(index);
    return o;
}

float4 _cubemaparray_frag_main(thread const cube_vert_output& _entryPointOutput, constant DebugConstants& v_230, thread texture2d_array<float> cubemap, thread sampler samp0)
{
    float3 param = _entryPointOutput.uvw;
    float3 uvw = convert_xyz_to_cube_uv(param);
    uvw.z += (v_230.layer_index * 6.0);
    return cubemap.sample(samp0, uvw.xy, uint(round(uvw.z)), level(v_230.mip_level));
}

fragment cubemaparray_frag_main_out cubemaparray_frag_main(cubemaparray_frag_main_in in [[stage_in]], constant DebugConstants& v_230 [[buffer(13)]], texture2d_array<float> cubemap [[texture(0)]], sampler samp0 [[sampler(0)]], float4 gl_FragCoord [[position]])
{
    cubemaparray_frag_main_out out = {};
    cube_vert_output _entryPointOutput;
    _entryPointOutput.pos = gl_FragCoord;
    _entryPointOutput.uvw = in._entryPointOutput_uvw;
    cube_vert_output param = _entryPointOutput;
    out._entryPointOutput = _cubemaparray_frag_main(param, v_230, cubemap, samp0);
    return out;
}

