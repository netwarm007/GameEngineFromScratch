#version 300 es
precision mediump float;
precision highp int;

struct cube_vert_output
{
    highp vec4 pos;
    highp vec3 uvw;
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

uniform highp sampler2DArray SPIRV_Cross_Combinedskyboxsamp0;

in highp vec3 _entryPointOutput_uvw;
layout(location = 0) out highp vec4 _entryPointOutput;

highp vec3 convert_xyz_to_cube_uv(highp vec3 d)
{
    highp vec3 d_abs = abs(d);
    bvec3 isPositive;
    isPositive.x = int(d.x > 0.0) != int(0u);
    isPositive.y = int(d.y > 0.0) != int(0u);
    isPositive.z = int(d.z > 0.0) != int(0u);
    highp float maxAxis;
    highp float uc;
    highp float vc;
    int index;
    if ((isPositive.x && (d_abs.x >= d_abs.y)) && (d_abs.x >= d_abs.z))
    {
        maxAxis = d_abs.x;
        uc = -d.z;
        vc = d.y;
        index = 0;
    }
    if (((!isPositive.x) && (d_abs.x >= d_abs.y)) && (d_abs.x >= d_abs.z))
    {
        maxAxis = d_abs.x;
        uc = d.z;
        vc = d.y;
        index = 1;
    }
    if ((isPositive.y && (d_abs.y >= d_abs.x)) && (d_abs.y >= d_abs.z))
    {
        maxAxis = d_abs.y;
        uc = d.x;
        vc = -d.z;
        index = 3;
    }
    if (((!isPositive.y) && (d_abs.y >= d_abs.x)) && (d_abs.y >= d_abs.z))
    {
        maxAxis = d_abs.y;
        uc = d.x;
        vc = d.z;
        index = 2;
    }
    if ((isPositive.z && (d_abs.z >= d_abs.x)) && (d_abs.z >= d_abs.y))
    {
        maxAxis = d_abs.z;
        uc = d.x;
        vc = d.y;
        index = 4;
    }
    if (((!isPositive.z) && (d_abs.z >= d_abs.x)) && (d_abs.z >= d_abs.y))
    {
        maxAxis = d_abs.z;
        uc = -d.x;
        vc = d.y;
        index = 5;
    }
    highp vec3 o;
    o.x = 0.5 * ((uc / maxAxis) + 1.0);
    o.y = 0.5 * ((vc / maxAxis) + 1.0);
    o.z = float(index);
    return o;
}

highp vec3 exposure_tone_mapping(highp vec3 color)
{
    return vec3(1.0) - exp((-color) * 1.0);
}

highp vec3 gamma_correction(highp vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(0.4545454680919647216796875));
}

highp vec4 _skybox_frag_main(cube_vert_output _entryPointOutput_1)
{
    highp vec3 param = _entryPointOutput_1.uvw;
    highp vec3 uvw = convert_xyz_to_cube_uv(param);
    highp vec4 outputColor = textureLod(SPIRV_Cross_Combinedskyboxsamp0, uvw, 0.0);
    highp vec3 param_1 = outputColor.xyz;
    highp vec3 _267 = exposure_tone_mapping(param_1);
    outputColor = vec4(_267.x, _267.y, _267.z, outputColor.w);
    highp vec3 param_2 = outputColor.xyz;
    highp vec3 _273 = gamma_correction(param_2);
    outputColor = vec4(_273.x, _273.y, _273.z, outputColor.w);
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

