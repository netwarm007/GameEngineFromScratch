#version 400

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

layout(std140) uniform PerBatchConstants
{
    mat4 modelMatrix;
} _25;

struct constants_t
{
    mat4 depthVP;
};

uniform constants_t u_pushConstants;

layout(location = 0) in vec3 inputPosition;

void main()
{
    gl_Position = (u_pushConstants.depthVP * _25.modelMatrix) * vec4(inputPosition, 1.0);
}

