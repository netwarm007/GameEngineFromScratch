#version 300 es
precision mediump float;
precision highp int;

struct simple_vert_output
{
    highp vec4 pos;
    highp vec2 uv;
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

layout(std140) uniform DebugConstants
{
    highp float layer_index;
    highp float mip_level;
    highp float line_width;
    highp float padding0;
    highp vec4 front_color;
    highp vec4 back_color;
} _32;

uniform highp sampler2DArray SPIRV_Cross_Combinedtexture_arraysamp0;

in highp vec2 _entryPointOutput_uv;
layout(location = 0) out highp vec4 _entryPointOutput;

highp vec4 _texturearray_frag_main(simple_vert_output _entryPointOutput_1)
{
    return textureLod(SPIRV_Cross_Combinedtexture_arraysamp0, vec3(_entryPointOutput_1.uv, _32.layer_index), _32.mip_level);
}

void main()
{
    simple_vert_output _entryPointOutput_1;
    _entryPointOutput_1.pos = gl_FragCoord;
    _entryPointOutput_1.uv = _entryPointOutput_uv;
    simple_vert_output param = _entryPointOutput_1;
    _entryPointOutput = _texturearray_frag_main(param);
}

