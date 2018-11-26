#version 400

struct a2v
{
    vec3 inputPosition;
    vec2 inputUV;
    vec3 inputNormal;
    vec3 inputTangent;
    vec3 inputBiTangent;
};

struct debug_vert_output
{
    vec4 position;
};

struct Light
{
    float lightIntensity;
    uint lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    uint lightAngleAttenCurveType;
    uint lightDistAttenCurveType;
    vec2 lightSize;
    uvec4 lightGuid;
    vec4 lightPosition;
    vec4 lightColor;
    vec4 lightDirection;
    vec4 lightDistAttenCurveParams[2];
    vec4 lightAngleAttenCurveParams[2];
    mat4 lightVP;
    vec4 padding[2];
};

layout(std140) uniform PerFrameConstants
{
    layout(row_major) mat4 viewMatrix;
    layout(row_major) mat4 projectionMatrix;
    vec4 camPos;
    uint numLights;
    float padding[3];
    Light lights[100];
} _43;

layout(location = 0) in vec3 a_inputPosition;
layout(location = 1) in vec2 a_inputUV;
layout(location = 2) in vec3 a_inputNormal;
layout(location = 3) in vec3 a_inputTangent;
layout(location = 4) in vec3 a_inputBiTangent;

debug_vert_output _debug_vert_main(a2v a)
{
    vec4 v = vec4(a.inputPosition, 1.0);
    v = _43.viewMatrix * v;
    debug_vert_output o;
    o.position = _43.projectionMatrix * v;
    return o;
}

void main()
{
    a2v a;
    a.inputPosition = a_inputPosition;
    a.inputUV = a_inputUV;
    a.inputNormal = a_inputNormal;
    a.inputTangent = a_inputTangent;
    a.inputBiTangent = a_inputBiTangent;
    a2v param = a;
    gl_Position = _debug_vert_main(param).position;
}

