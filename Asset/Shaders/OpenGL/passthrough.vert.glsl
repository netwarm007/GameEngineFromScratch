#version 420

struct a2v_simple
{
    vec3 inputPosition;
    vec2 inputUV;
};

struct simple_vert_output
{
    vec4 pos;
    vec2 uv;
};

struct Light
{
    float lightIntensity;
    int lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    vec2 lightSize;
    ivec4 lightGuid;
    vec4 lightPosition;
    vec4 lightColor;
    vec4 lightDirection;
    vec4 lightDistAttenCurveParams[2];
    vec4 lightAngleAttenCurveParams[2];
    mat4 lightVP;
    vec4 padding[2];
};

layout(location = 0) in vec3 a_inputPosition;
layout(location = 1) in vec2 a_inputUV;
layout(location = 0) out vec2 _entryPointOutput_uv;

simple_vert_output _passthrough_vert_main(a2v_simple a)
{
    simple_vert_output o;
    o.pos = vec4(a.inputPosition, 1.0);
    o.uv = a.inputUV;
    return o;
}

void main()
{
    a2v_simple a;
    a.inputPosition = a_inputPosition;
    a.inputUV = a_inputUV;
    a2v_simple param = a;
    simple_vert_output flattenTemp = _passthrough_vert_main(param);
    gl_Position = flattenTemp.pos;
    _entryPointOutput_uv = flattenTemp.uv;
}

