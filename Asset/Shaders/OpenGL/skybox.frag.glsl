#version 420

struct cube_vert_output
{
    vec4 pos;
    vec3 uvw;
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

uniform samplerCubeArray SPIRV_Cross_Combinedskyboxsamp0;

layout(location = 0) in vec3 _entryPointOutput_uvw;
layout(location = 0) out vec4 _entryPointOutput;

vec3 exposure_tone_mapping(vec3 color)
{
    return vec3(1.0) - exp((-color) * 1.0);
}

vec3 gamma_correction(vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(0.4545454680919647216796875));
}

vec4 _skybox_frag_main(cube_vert_output _entryPointOutput_1)
{
    vec4 outputColor = textureLod(SPIRV_Cross_Combinedskyboxsamp0, vec4(_entryPointOutput_1.uvw, 0.0), 0.0);
    vec3 param = outputColor.xyz;
    vec3 _65 = exposure_tone_mapping(param);
    outputColor = vec4(_65.x, _65.y, _65.z, outputColor.w);
    vec3 param_1 = outputColor.xyz;
    vec3 _71 = gamma_correction(param_1);
    outputColor = vec4(_71.x, _71.y, _71.z, outputColor.w);
    return outputColor;
}

void main()
{
    cube_vert_output _entryPointOutput_1;
    _entryPointOutput_1.pos = gl_FragCoord;
    _entryPointOutput_1.uvw = _entryPointOutput_uvw;
    cube_vert_output param = _entryPointOutput_1;
    _entryPointOutput = _skybox_frag_main(param);
}

