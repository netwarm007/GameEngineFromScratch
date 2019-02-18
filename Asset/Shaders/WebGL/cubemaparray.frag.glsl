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

layout(std140) uniform DebugConstants
{
    highp float layer_index;
    highp float mip_level;
    highp float line_width;
    highp float padding0;
    highp vec4 front_color;
    highp vec4 back_color;
} _230;

uniform highp sampler2DArray SPIRV_Cross_Combinedcubemapsamp0;

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

highp vec4 _cubemaparray_frag_main(cube_vert_output _entryPointOutput_1)
{
    highp vec3 param = _entryPointOutput_1.uvw;
    highp vec3 uvw = convert_xyz_to_cube_uv(param);
    uvw.z += (_230.layer_index * 6.0);
    return textureLod(SPIRV_Cross_Combinedcubemapsamp0, uvw, _230.mip_level);
}

void main()
{
    cube_vert_output _entryPointOutput_1;
    _entryPointOutput_1.pos = gl_FragCoord;
    _entryPointOutput_1.uvw = _entryPointOutput_uvw;
    cube_vert_output param = _entryPointOutput_1;
    _entryPointOutput = _cubemaparray_frag_main(param);
}

