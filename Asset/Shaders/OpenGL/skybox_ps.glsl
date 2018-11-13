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

uniform samplerCubeArray skybox;

layout(location = 0) out vec4 outputColor;
in vec3 UVW;

vec3 exposure_tone_mapping(vec3 color)
{
    return vec3(1.0) - exp((-color) * 1.0);
}

vec3 gamma_correction(vec3 color)
{
    return pow(color, vec3(0.4545454680919647216796875));
}

void main()
{
    outputColor = textureLod(skybox, vec4(UVW, 0.0), 0.0);
    vec3 param = outputColor.xyz;
    vec3 _51 = exposure_tone_mapping(param);
    outputColor = vec4(_51.x, _51.y, _51.z, outputColor.w);
    vec3 param_1 = outputColor.xyz;
    vec3 _57 = gamma_correction(param_1);
    outputColor = vec4(_57.x, _57.y, _57.z, outputColor.w);
}

