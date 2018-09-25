#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Light
{
    float lightIntensity;
    int lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    float2 lightSize;
    int4 lightGUID;
    float4 lightPosition;
    float4 lightColor;
    float4 lightDirection;
    float4 lightDistAttenCurveParams[2];
    float4 lightAngleAttenCurveParams[2];
    float4x4 lightVP;
    float4 padding[2];
};

struct PerFrameConstants
{
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4 camPos;
    int numLights;
    Light allLights[100];
};

struct PerBatchConstants
{
    float4x4 modelMatrix;
};

struct main0_out
{
    float4 outputColor [[color(0)]];
};

struct main0_in
{
    float3 UVW [[user(locn0)]];
};

float3 inverse_gamma_correction(thread const float3& color)
{
    return pow(color, float3(2.2000000476837158203125));
}

float3 exposure_tone_mapping(thread const float3& color)
{
    return float3(1.0) - exp((-color) * 1.0);
}

float3 gamma_correction(thread const float3& color)
{
    return pow(color, float3(0.4545454680919647216796875));
}

fragment main0_out main0(main0_in in [[stage_in]], texturecube_array<float> skybox [[texture(4)]], sampler skyboxSmplr [[sampler(4)]])
{
    main0_out out = {};
    out.outputColor = skybox.sample(skyboxSmplr, float4(in.UVW, 0.0).xyz, uint(round(float4(in.UVW, 0.0).w)), level(0.0));
    float3 param = out.outputColor.xyz;
    float3 _60 = inverse_gamma_correction(param);
    out.outputColor = float4(_60.x, _60.y, _60.z, out.outputColor.w);
    float3 param_1 = out.outputColor.xyz;
    float3 _66 = exposure_tone_mapping(param_1);
    out.outputColor = float4(_66.x, _66.y, _66.z, out.outputColor.w);
    float3 param_2 = out.outputColor.xyz;
    float3 _72 = gamma_correction(param_2);
    out.outputColor = float4(_72.x, _72.y, _72.z, out.outputColor.w);
    return out;
}

