#version 400

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

struct debugPushConstants
{
    float level;
    float layer_index;
};

uniform debugPushConstants u_pushConstants;

uniform samplerCubeArray depthSampler;

layout(location = 0) out vec3 color;
in vec3 UVW;

void main()
{
    color = textureLod(depthSampler, vec4(UVW, u_pushConstants.layer_index), u_pushConstants.level).xxx;
}

