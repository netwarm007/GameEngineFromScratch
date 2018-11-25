#version 400

struct pos_only_vert_output
{
    vec4 pos;
};

struct Light
{
    float lightIntensity;
    uint lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    uint lightAngleAttenCurveType;
    uint lightDistAttenCurveType;
    vec2 lightSize;
    uvec4 lightGuid;
    vec4 lightPosition;
    vec4 lightColor;
    vec4 lightDirection;
    vec4 lightDistAttenCurveParams[2];
    vec4 lightAngleAttenCurveParams[2];
    mat4 lightVP;
    vec4 padding[2];
};

layout(location = 0) out vec4 _entryPointOutput;

vec4 _debug_frag_main(pos_only_vert_output _input)
{
    return vec4(1.0);
}

void main()
{
    pos_only_vert_output _input;
    _input.pos = gl_FragCoord;
    pos_only_vert_output param = _input;
    _entryPointOutput = _debug_frag_main(param);
}

