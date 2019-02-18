#version 320 es

struct a2v_pos_only
{
    vec3 inputPosition;
};

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

layout(binding = 11, std140) uniform PerBatchConstants
{
    mat4 modelMatrix;
} _31;

layout(binding = 14, std140) uniform ShadowMapConstants
{
    mat4 shadowMatrices[6];
    vec4 lightPos;
    float shadowmap_layer_index;
    float far_plane;
} _44;

layout(location = 0) in vec3 a_inputPosition;

pos_only_vert_output _shadowmap_vert_main(a2v_pos_only a)
{
    vec4 v = vec4(a.inputPosition, 1.0);
    v = _31.modelMatrix * v;
    pos_only_vert_output o;
    o.pos = _44.shadowMatrices[0] * v;
    return o;
}

void main()
{
    a2v_pos_only a;
    a.inputPosition = a_inputPosition;
    a2v_pos_only param = a;
    gl_Position = _shadowmap_vert_main(param).pos;
}

