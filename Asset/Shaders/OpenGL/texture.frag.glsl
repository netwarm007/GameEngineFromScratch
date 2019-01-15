#version 420

struct simple_vert_output
{
    vec4 pos;
    vec2 uv;
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

uniform sampler2D SPIRV_Cross_Combinedtexsamp0;

layout(location = 0) in vec2 _entryPointOutput_uv;
layout(location = 0) out vec4 _entryPointOutput;

vec4 _texture_frag_main(simple_vert_output _entryPointOutput_1)
{
    return texture(SPIRV_Cross_Combinedtexsamp0, _entryPointOutput_1.uv);
}

void main()
{
    simple_vert_output _entryPointOutput_1;
    _entryPointOutput_1.pos = gl_FragCoord;
    _entryPointOutput_1.uv = _entryPointOutput_uv;
    simple_vert_output param = _entryPointOutput_1;
    _entryPointOutput = _texture_frag_main(param);
}

