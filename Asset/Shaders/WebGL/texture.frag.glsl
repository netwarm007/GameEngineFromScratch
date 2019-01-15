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

uniform highp sampler2D SPIRV_Cross_Combinedtexsamp0;

in highp vec2 _entryPointOutput_uv;
layout(location = 0) out highp vec4 _entryPointOutput;

highp vec4 _texture_frag_main(simple_vert_output _entryPointOutput_1)
{
    return texture(SPIRV_Cross_Combinedtexsamp0, _entryPointOutput_1.uv);
}

void main()
{
    simple_vert_output _entryPointOutput_1;
    _entryPointOutput_1.pos = gl_FragCoord;
    _entryPointOutput_1.uv = _entryPointOutput_uv;
    simple_vert_output param = _entryPointOutput_1;
    _entryPointOutput = _texture_frag_main(param);
}

