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

layout(binding = 13, std140) uniform DebugConstants
{
    float layer_index;
    float mip_level;
    float line_width;
    float padding0;
    vec4 front_color;
    vec4 back_color;
} _32;

uniform sampler2DArray SPIRV_Cross_Combinedtexture_arraysamp0;

layout(location = 0) in vec2 _entryPointOutput_uv;
layout(location = 0) out vec4 _entryPointOutput;

vec4 _texturearray_frag_main(simple_vert_output _entryPointOutput_1)
{
    return textureLod(SPIRV_Cross_Combinedtexture_arraysamp0, vec3(_entryPointOutput_1.uv, _32.layer_index), _32.mip_level);
}

void main()
{
    simple_vert_output _entryPointOutput_1;
    _entryPointOutput_1.pos = gl_FragCoord;
    _entryPointOutput_1.uv = _entryPointOutput_uv;
    simple_vert_output param = _entryPointOutput_1;
    _entryPointOutput = _texturearray_frag_main(param);
}

