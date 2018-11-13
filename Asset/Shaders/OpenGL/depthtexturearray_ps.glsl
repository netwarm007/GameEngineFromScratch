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
    float layer_index;
};

uniform debugPushConstants u_pushConstants;

uniform sampler2DArray depthSampler;

layout(location = 0) out vec3 color;
in vec2 UV;

void main()
{
    color = texture(depthSampler, vec3(UV, u_pushConstants.layer_index)).xxx;
}

