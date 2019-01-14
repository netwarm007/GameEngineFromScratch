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

layout(location = 0) out vec4 _entryPointOutput;

vec4 _debug_frag_main(pos_only_vert_output _entryPointOutput_1)
{
    return vec4(1.0);
}

void main()
{
    pos_only_vert_output _entryPointOutput_1;
    _entryPointOutput_1.pos = gl_FragCoord;
    pos_only_vert_output param = _entryPointOutput_1;
    _entryPointOutput = _debug_frag_main(param);
}

