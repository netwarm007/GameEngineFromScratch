#version 310 es
precision mediump float;
precision highp int;

struct pos_only_vert_output
{
    highp vec4 pos;
};

struct Light
{
    highp float lightIntensity;
    uint lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    uint lightAngleAttenCurveType;
    uint lightDistAttenCurveType;
    highp vec2 lightSize;
    uvec4 lightGuid;
    highp vec4 lightPosition;
    highp vec4 lightColor;
    highp vec4 lightDirection;
    highp vec4 lightDistAttenCurveParams[2];
    highp vec4 lightAngleAttenCurveParams[2];
    highp mat4 lightVP;
    highp vec4 padding[2];
};

layout(location = 0) out highp vec4 _entryPointOutput;

highp vec4 _debug_frag_main(pos_only_vert_output _input)
{
    return vec4(1.0);
}

void main()
{
    pos_only_vert_output _input;
    _input.pos = gl_FragCoord;
    pos_only_vert_output param = _input;
    _entryPointOutput = _debug_frag_main(param);
}

