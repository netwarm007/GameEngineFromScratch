#version 320 es
precision mediump float;
precision highp int;

struct cube_vert_output
{
    highp vec4 pos;
    highp vec3 uvw;
};

struct Light
{
    highp float lightIntensity;
    int lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    highp vec2 lightSize;
    ivec4 lightGuid;
    highp vec4 lightPosition;
    highp vec4 lightColor;
    highp vec4 lightDirection;
    highp vec4 lightDistAttenCurveParams[2];
    highp vec4 lightAngleAttenCurveParams[2];
    highp mat4 lightVP;
    highp vec4 padding[2];
};

layout(binding = 13, std140) uniform DebugConstants
{
    highp float layer_index;
    highp float mip_level;
    highp float line_width;
    highp float padding0;
    highp vec4 front_color;
    highp vec4 back_color;
} _32;

uniform highp samplerCubeArray SPIRV_Cross_Combinedcubemapsamp0;

layout(location = 0) in highp vec3 _entryPointOutput_uvw;
layout(location = 0) out highp vec4 _entryPointOutput;

highp vec4 _cubemaparray_frag_main(cube_vert_output _entryPointOutput_1)
{
    return textureLod(SPIRV_Cross_Combinedcubemapsamp0, vec4(_entryPointOutput_1.uvw, _32.layer_index), _32.mip_level);
}

void main()
{
    cube_vert_output _entryPointOutput_1;
    _entryPointOutput_1.pos = gl_FragCoord;
    _entryPointOutput_1.uvw = _entryPointOutput_uvw;
    cube_vert_output param = _entryPointOutput_1;
    _entryPointOutput = _cubemaparray_frag_main(param);
}

