#version 420

struct pos_only_vert_output
{
    vec4 pos;
};

struct Light
{
    float lightIntensity;
    int lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    vec2 lightSize;
    ivec4 lightGuid;
    vec4 lightPosition;
    vec4 lightColor;
    vec4 lightDirection;
    vec4 lightDistAttenCurveParams[2];
    vec4 lightAngleAttenCurveParams[2];
    mat4 lightVP;
    vec4 padding[2];
};

layout(binding = 14, std140) uniform ShadowMapConstants
{
    mat4 shadowMatrices[6];
    vec4 lightPos;
    float shadowmap_layer_index;
    float far_plane;
} _29;

float _shadowmap_omni_frag_main(pos_only_vert_output _entryPointOutput)
{
    float lightDistance = length(_entryPointOutput.pos.xyz - vec3(_29.lightPos.xyz));
    lightDistance /= _29.far_plane;
    return lightDistance;
}

void main()
{
    pos_only_vert_output _entryPointOutput;
    _entryPointOutput.pos = gl_FragCoord;
    pos_only_vert_output param = _entryPointOutput;
    gl_FragDepth = _shadowmap_omni_frag_main(param);
}

