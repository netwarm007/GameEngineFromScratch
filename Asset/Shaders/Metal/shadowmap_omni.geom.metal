#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct pos_only_vert_output
{
    float4 pos;
};

struct gs_layered_output
{
    float4 pos;
    int slice;
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

struct shadowmap_omni_geom_main_out
{
    float4 gl_Position;
    int gl_Layer;
};

// Implementation of an array copy function to cover GLSL's ability to copy an array via assignment.
template<typename T, uint N>
void spvArrayCopy(thread T (&dst)[N], thread const T (&src)[N])
{
    for (uint i = 0; i < N; dst[i] = src[i], i++);
}

// An overload for constant arrays.
template<typename T, uint N>
void spvArrayCopyConstant(thread T (&dst)[N], constant T (&src)[N])
{
    for (uint i = 0; i < N; dst[i] = src[i], i++);
}

void _shadowmap_omni_geom_main(thread const pos_only_vert_output (&_entryPointOutput)[3], thread const gs_layered_output& OutputStream, constant ShadowMapConstants& v_40, thread float4& gl_Position, thread uint& gl_Layer)
{
    for (int face = 0; face < 6; face++)
    {
        gs_layered_output _output;
        _output.slice = (int(v_40.shadowmap_layer_index) * 6) + face;
        for (int i = 0; i < 3; i++)
        {
            _output.pos = v_40.shadowMatrices[face] * _entryPointOutput[i].pos;
            gl_Position = _output.pos;
            gl_Layer = _output.slice;
            EmitVertex();
        }
        EndPrimitive();
    }
}

unknown shadowmap_omni_geom_main_out shadowmap_omni_geom_main(constant ShadowMapConstants& v_40 [[buffer(14)]], float4 gl_Position [[position]])
{
    shadowmap_omni_geom_main_out out = {};
    pos_only_vert_output _entryPointOutput[3];
    _entryPointOutput[0].pos = gl_Position[0];
    _entryPointOutput[1].pos = gl_Position[1];
    _entryPointOutput[2].pos = gl_Position[2];
    pos_only_vert_output param[3];
    spvArrayCopy(param, _entryPointOutput);
    gs_layered_output param_1;
    _shadowmap_omni_geom_main(param, param_1, v_40, out.gl_Position, out.gl_Layer);
    gs_layered_output OutputStream = param_1;
    return out;
}

