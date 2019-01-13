#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct cube_vert_output
{
    float4 pos;
    float3 uvw;
};

struct PerFrameConstants
{
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 arbitraryMatrix;
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

struct skybox_frag_main_out
{
    float4 _entryPointOutput [[color(0)]];
};

struct skybox_frag_main_in
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

float3 exposure_tone_mapping(thread const float3& color)
{
    return float3(1.0) - exp((-color) * 1.0);
}

float3 gamma_correction(thread const float3& color)
{
    return pow(max(color, float3(0.0)), float3(0.4545454680919647216796875));
}

float4 _skybox_frag_main(thread const cube_vert_output& _entryPointOutput, thread texture2d_array<float> skybox, thread sampler samp0)
{
    float3 param = _entryPointOutput.uvw;
    float3 uvw = convert_xyz_to_cube_uv(param);
    float4 outputColor = skybox.sample(samp0, uvw.xy, uint(round(uvw.z)), level(0.0));
    float3 param_1 = outputColor.xyz;
    float3 _267 = exposure_tone_mapping(param_1);
    outputColor = float4(_267.x, _267.y, _267.z, outputColor.w);
    float3 param_2 = outputColor.xyz;
    float3 _273 = gamma_correction(param_2);
    outputColor = float4(_273.x, _273.y, _273.z, outputColor.w);
    return outputColor;
}

fragment skybox_frag_main_out skybox_frag_main(skybox_frag_main_in in [[stage_in]], texture2d_array<float> skybox [[texture(10)]], sampler samp0 [[sampler(0)]], float4 gl_FragCoord [[position]])
{
    skybox_frag_main_out out = {};
    cube_vert_output _entryPointOutput;
    _entryPointOutput.pos = gl_FragCoord;
    _entryPointOutput.uvw = in._entryPointOutput_uvw;
    cube_vert_output param = _entryPointOutput;
    out._entryPointOutput = _skybox_frag_main(param, skybox, samp0);
    return out;
}

