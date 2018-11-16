#version 310 es
precision mediump float;
precision highp int;

struct vert_output
{
    highp vec4 position;
    highp vec4 normal;
    highp vec4 normal_world;
    highp vec4 v;
    highp vec4 v_world;
    highp vec2 uv;
    highp mat3 TBN;
    highp vec3 v_tangent;
    highp vec3 camPos_tangent;
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

layout(location = 0) in highp vec4 input_normal;
layout(location = 1) in highp vec4 input_normal_world;
layout(location = 2) in highp vec4 input_v;
layout(location = 3) in highp vec4 input_v_world;
layout(location = 4) in highp vec2 input_uv;
layout(location = 5) in highp mat3 input_TBN;
layout(location = 8) in highp vec3 input_v_tangent;
layout(location = 9) in highp vec3 input_camPos_tangent;
layout(location = 0) out highp vec4 _entryPointOutput;

highp vec4 _debug_frag_main(vert_output _input)
{
    return vec4(1.0);
}

void main()
{
    vert_output _input;
    _input.position = gl_FragCoord;
    _input.normal = input_normal;
    _input.normal_world = input_normal_world;
    _input.v = input_v;
    _input.v_world = input_v_world;
    _input.uv = input_uv;
    _input.TBN = input_TBN;
    _input.v_tangent = input_v_tangent;
    _input.camPos_tangent = input_camPos_tangent;
    vert_output param = _input;
    _entryPointOutput = _debug_frag_main(param);
}

