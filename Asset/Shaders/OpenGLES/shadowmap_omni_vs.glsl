#version 310 es

struct Light
{
    float lightIntensity;
    int lightType;
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

layout(binding = 1, std140) uniform PerBatchConstants
{
    mat4 modelMatrix;
} _19;

layout(location = 0) in vec3 inputPosition;

void main()
{
    gl_Position = _19.modelMatrix * vec4(inputPosition, 1.0);
}

