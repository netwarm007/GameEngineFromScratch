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

struct debugPushConstants
{
    vec3 FrontColor;
};

uniform debugPushConstants u_pushConstants;

layout(location = 0) out vec4 outputColor;

void main()
{
    outputColor = vec4(u_pushConstants.FrontColor, 1.0);
}

