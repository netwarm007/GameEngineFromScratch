#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct a2v
{
    float3 inputPosition;
    float3 inputNormal;
    float3 inputUV;
    float3 inputTangent;
    float3 inputBiTangent;
};

struct debug_vert_output
{
    float4 position;
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

struct PerFrameConstants
{
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4 camPos;
    uint numLights;
    char pad4[12];
    float padding[3];
    Light lights[100];
};

struct PerBatchConstants
{
    float4x4 modelMatrix;
};

struct debug_vert_main_out
{
    float4 gl_Position [[position]];
};

struct debug_vert_main_in
{
    float3 a_inputPosition [[attribute(0)]];
    float3 a_inputNormal [[attribute(1)]];
    float3 a_inputUV [[attribute(2)]];
    float3 a_inputTangent [[attribute(3)]];
    float3 a_inputBiTangent [[attribute(4)]];
};

debug_vert_output _debug_vert_main(thread const a2v& a, constant PerFrameConstants& v_43)
{
    float4 v = float4(a.inputPosition, 1.0);
    v *= v_43.viewMatrix;
    debug_vert_output o;
    o.position = v * v_43.projectionMatrix;
    return o;
}

vertex debug_vert_main_out debug_vert_main(debug_vert_main_in in [[stage_in]], constant PerFrameConstants& v_43 [[buffer(0)]])
{
    debug_vert_main_out out = {};
    a2v a;
    a.inputPosition = in.a_inputPosition;
    a.inputNormal = in.a_inputNormal;
    a.inputUV = in.a_inputUV;
    a.inputTangent = in.a_inputTangent;
    a.inputBiTangent = in.a_inputBiTangent;
    a2v param = a;
    out.gl_Position = _debug_vert_main(param, v_43).position;
    return out;
}

