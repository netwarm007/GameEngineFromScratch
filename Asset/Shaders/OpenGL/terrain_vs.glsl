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

uniform sampler2D terrainHeightMap;

layout(location = 0) in vec3 inputPosition;

void main()
{
    float height = textureLod(terrainHeightMap, inputPosition.xy, 0.0).x;
    vec4 displaced = vec4(inputPosition.xy, height, 1.0);
    gl_Position = displaced;
}

