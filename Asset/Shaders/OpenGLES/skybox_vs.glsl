#version 310 es

struct Light
{
    int lightType;
    float lightIntensity;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    vec2 lightSize;
    ivec4 lightGUID;
    vec4 lightPosition;
    vec4 lightColor;
    vec4 lightDirection;
    vec4 lightDistAttenCurveParams[2];
    vec4 lightAngleAttenCurveParams[2];
    mat4 lightVP;
    vec4 padding[2];
};

layout(binding = 0, std140) uniform PerFrameConstants
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 camPos;
    int numLights;
    Light allLights[100];
} _30;

layout(location = 0) out vec3 UVW;
layout(location = 0) in vec3 inputPosition;

void main()
{
    UVW = inputPosition;
    mat4 matrix = _30.viewMatrix;
    matrix[3].x = 0.0;
    matrix[3].y = 0.0;
    matrix[3].z = 0.0;
    vec4 pos = (_30.projectionMatrix * matrix) * vec4(inputPosition, 1.0);
    gl_Position = pos.xyww;
}

