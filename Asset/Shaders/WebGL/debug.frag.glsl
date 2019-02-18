#version 300 es
precision mediump float;
precision highp int;

struct pos_only_vert_output
{
    highp vec4 pos;
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

layout(location = 0) out highp vec4 _entryPointOutput;

highp vec4 _debug_frag_main(pos_only_vert_output _entryPointOutput_1)
{
    return vec4(1.0);
}

void main()
{
    pos_only_vert_output _entryPointOutput_1;
    _entryPointOutput_1.pos = gl_FragCoord;
    pos_only_vert_output param = _entryPointOutput_1;
    _entryPointOutput = _debug_frag_main(param);
}

