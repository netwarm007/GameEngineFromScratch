#version 300 es
precision mediump float;
precision highp int;

struct pos_only_vert_output
{
    highp vec4 pos;
};

struct Light
{
    highp float lightIntensity;
    int lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    highp vec2 lightSize;
    ivec4 lightGuid;
    highp vec4 lightPosition;
    highp vec4 lightColor;
    highp vec4 lightDirection;
    highp vec4 lightDistAttenCurveParams[2];
    highp vec4 lightAngleAttenCurveParams[2];
    highp mat4 lightVP;
    highp vec4 padding[2];
};

layout(std140) uniform ShadowMapConstants
{
    highp mat4 shadowMatrices[6];
    highp vec4 lightPos;
    highp float shadowmap_layer_index;
    highp float far_plane;
} _29;

highp float _shadowmap_omni_frag_main(pos_only_vert_output _entryPointOutput)
{
    highp float lightDistance = length(_entryPointOutput.pos.xyz - vec3(_29.lightPos.xyz));
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

