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

layout(location = 0) in vec3 inputPosition;
layout(location = 0) out vec2 UV;
layout(location = 1) in vec2 inputUV;

void main()
{
    gl_Position = vec4(inputPosition, 1.0);
    UV = inputUV;
}

