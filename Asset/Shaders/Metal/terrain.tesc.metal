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

struct terrain_tesc_main_out
{
    float4 gl_Position;
    float gl_TessLevelInner[2];
    float gl_TessLevelOuter[4];
};

float4 project(thread const float4& vertex0, constant PerFrameConstants& v_43)
{
    float4 result = (v_43.projectionMatrix * v_43.viewMatrix) * vertex0;
    result /= float4(result.w);
    return result;
}

bool offscreen(thread const float4& vertex0)
{
    if (vertex0.z < (-0.5))
    {
        return true;
    }
    bool _94 = any(vertex0.xy < float2(-1.7000000476837158203125));
    bool _104;
    if (!_94)
    {
        _104 = any(vertex0.xy > float2(1.7000000476837158203125));
    }
    else
    {
        _104 = _94;
    }
    return _104;
}

float2 screen_space(thread const float4& vertex0)
{
    return (clamp(vertex0.xy, float2(-1.2999999523162841796875), float2(1.2999999523162841796875)) + float2(1.0)) * float2(480.0, 270.0);
}

float level(thread const float2& v0, thread const float2& v1)
{
    return clamp(distance(v0, v1) / 2.0, 1.0, 64.0);
}

unknown terrain_tesc_main_out terrain_tesc_main(constant PerFrameConstants& v_43 [[buffer(1)]], unsupported-built-in-type gl_InvocationID [[unsupported-built-in]], float4 gl_in [[position]])
{
    terrain_tesc_main_out out = {};
    gl_out[gl_InvocationID].out.gl_Position = gl_in[gl_InvocationID].out.gl_Position;
    if (gl_InvocationID == 0)
    {
        float4 param = gl_in[0].out.gl_Position;
        float4 v0 = project(param, v_43);
        float4 param_1 = gl_in[1].out.gl_Position;
        float4 v1 = project(param_1, v_43);
        float4 param_2 = gl_in[2].out.gl_Position;
        float4 v2 = project(param_2, v_43);
        float4 param_3 = gl_in[3].out.gl_Position;
        float4 v3 = project(param_3, v_43);
        float4 param_4 = v0;
        float4 param_5 = v1;
        float4 param_6 = v2;
        float4 param_7 = v3;
        if (all(bool4(offscreen(param_4), offscreen(param_5), offscreen(param_6), offscreen(param_7))))
        {
            gl_TessLevelInner[0] = 0.0;
            gl_TessLevelInner[1] = 0.0;
            gl_TessLevelOuter[0] = 0.0;
            gl_TessLevelOuter[1] = 0.0;
            gl_TessLevelOuter[2] = 0.0;
            gl_TessLevelOuter[3] = 0.0;
        }
        else
        {
            float4 param_8 = v0;
            float2 ss0 = screen_space(param_8);
            float4 param_9 = v1;
            float2 ss1 = screen_space(param_9);
            float4 param_10 = v2;
            float2 ss2 = screen_space(param_10);
            float4 param_11 = v3;
            float2 ss3 = screen_space(param_11);
            float2 param_12 = ss1;
            float2 param_13 = ss2;
            float e0 = level(param_12, param_13);
            float2 param_14 = ss0;
            float2 param_15 = ss1;
            float e1 = level(param_14, param_15);
            float2 param_16 = ss3;
            float2 param_17 = ss0;
            float e2 = level(param_16, param_17);
            float2 param_18 = ss2;
            float2 param_19 = ss3;
            float e3 = level(param_18, param_19);
            gl_TessLevelInner[0] = mix(e1, e2, 0.5);
            gl_TessLevelInner[1] = mix(e0, e3, 0.5);
            gl_TessLevelOuter[0] = e0;
            gl_TessLevelOuter[1] = e1;
            gl_TessLevelOuter[2] = e2;
            gl_TessLevelOuter[3] = e3;
        }
    }
    return out;
}

