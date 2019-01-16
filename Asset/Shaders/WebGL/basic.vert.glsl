#version 300 es

struct a2v
{
    vec3 inputPosition;
    vec3 inputNormal;
    vec2 inputUV;
    vec3 inputTangent;
};

struct basic_vert_output
{
    vec4 pos;
    vec4 normal;
    vec4 normal_world;
    vec4 v;
    vec4 v_world;
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

layout(std140) uniform PerBatchConstants
{
    mat4 modelMatrix;
} _24;

layout(std140) uniform PerFrameConstants
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 camPos;
    int numLights;
} _43;

layout(location = 0) in vec3 a_inputPosition;
layout(location = 1) in vec3 a_inputNormal;
layout(location = 2) in vec2 a_inputUV;
layout(location = 3) in vec3 a_inputTangent;
out vec4 _entryPointOutput_normal;
out vec4 _entryPointOutput_normal_world;
out vec4 _entryPointOutput_v;
out vec4 _entryPointOutput_v_world;
out vec2 _entryPointOutput_uv;

basic_vert_output _basic_vert_main(a2v a)
{
    basic_vert_output o;
    o.v_world = _24.modelMatrix * vec4(a.inputPosition, 1.0);
    o.v = _43.viewMatrix * o.v_world;
    o.pos = _43.projectionMatrix * o.v;
    o.normal_world = normalize(_24.modelMatrix * vec4(a.inputNormal, 0.0));
    o.normal = normalize(_43.viewMatrix * o.normal_world);
    o.uv.x = a.inputUV.x;
    o.uv.y = 1.0 - a.inputUV.y;
    return o;
}

void main()
{
    a2v a;
    a.inputPosition = a_inputPosition;
    a.inputNormal = a_inputNormal;
    a.inputUV = a_inputUV;
    a.inputTangent = a_inputTangent;
    a2v param = a;
    basic_vert_output flattenTemp = _basic_vert_main(param);
    gl_Position = flattenTemp.pos;
    _entryPointOutput_normal = flattenTemp.normal;
    _entryPointOutput_normal_world = flattenTemp.normal_world;
    _entryPointOutput_v = flattenTemp.v;
    _entryPointOutput_v_world = flattenTemp.v_world;
    _entryPointOutput_uv = flattenTemp.uv;
}

