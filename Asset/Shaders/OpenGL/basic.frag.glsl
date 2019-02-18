#version 420

struct basic_vert_output
{
    vec4 pos;
    vec4 normal;
    vec4 normal_world;
    vec4 v;
    vec4 v_world;
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

layout(binding = 10, std140) uniform PerFrameConstants
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 camPos;
    int numLights;
} _280;

layout(binding = 12, std140) uniform LightInfo
{
    Light lights[100];
} _677;

uniform sampler2D SPIRV_Cross_CombineddiffuseMapsamp0;
uniform samplerCubeArray SPIRV_Cross_Combinedskyboxsamp0;

layout(location = 0) in vec4 _entryPointOutput_normal;
layout(location = 1) in vec4 _entryPointOutput_normal_world;
layout(location = 2) in vec4 _entryPointOutput_v;
layout(location = 3) in vec4 _entryPointOutput_v_world;
layout(location = 4) in vec2 _entryPointOutput_uv;
layout(location = 0) out vec4 _entryPointOutput;

float _130;

vec3 projectOnPlane(vec3 _point, vec3 center_of_plane, vec3 normal_of_plane)
{
    return _point - (normal_of_plane * dot(_point - center_of_plane, normal_of_plane));
}

float linear_interpolate(float t, float begin, float end)
{
    if (t < begin)
    {
        return 1.0;
    }
    else
    {
        if (t > end)
        {
            return 0.0;
        }
        else
        {
            return (end - t) / (end - begin);
        }
    }
}

float apply_atten_curve(float dist, int atten_curve_type, vec4 atten_params[2])
{
    float atten = 1.0;
    switch (atten_curve_type)
    {
        case 1:
        {
            float begin_atten = atten_params[0].x;
            float end_atten = atten_params[0].y;
            float param = dist;
            float param_1 = begin_atten;
            float param_2 = end_atten;
            float param_3 = param;
            float param_4 = param_1;
            float param_5 = param_2;
            atten = linear_interpolate(param_3, param_4, param_5);
            break;
        }
        case 2:
        {
            float begin_atten_1 = atten_params[0].x;
            float end_atten_1 = atten_params[0].y;
            float param_3_1 = dist;
            float param_4_1 = begin_atten_1;
            float param_5_1 = end_atten_1;
            float param_6 = param_3_1;
            float param_7 = param_4_1;
            float param_8 = param_5_1;
            float tmp = linear_interpolate(param_6, param_7, param_8);
            atten = (3.0 * pow(tmp, 2.0)) - (2.0 * pow(tmp, 3.0));
            break;
        }
        case 3:
        {
            float scale = atten_params[0].x;
            float offset = atten_params[0].y;
            float kl = atten_params[0].z;
            float kc = atten_params[0].w;
            atten = clamp((scale / ((kl * dist) + (kc * scale))) + offset, 0.0, 1.0);
            break;
        }
        case 4:
        {
            float scale_1 = atten_params[0].x;
            float offset_1 = atten_params[0].y;
            float kq = atten_params[0].z;
            float kl_1 = atten_params[0].w;
            float kc_1 = atten_params[1].x;
            atten = clamp(pow(scale_1, 2.0) / ((((kq * pow(dist, 2.0)) + ((kl_1 * dist) * scale_1)) + (kc_1 * pow(scale_1, 2.0))) + offset_1), 0.0, 1.0);
            break;
        }
        case 0:
        {
            break;
        }
        default:
        {
            break;
        }
    }
    return atten;
}

bool isAbovePlane(vec3 _point, vec3 center_of_plane, vec3 normal_of_plane)
{
    return dot(_point - center_of_plane, normal_of_plane) > 0.0;
}

vec3 linePlaneIntersect(vec3 line_start, vec3 line_dir, vec3 center_of_plane, vec3 normal_of_plane)
{
    return line_start + (line_dir * (dot(center_of_plane - line_start, normal_of_plane) / dot(line_dir, normal_of_plane)));
}

vec3 apply_areaLight(Light light, basic_vert_output _input)
{
    vec3 N = normalize(_input.normal.xyz);
    vec3 right = normalize((_280.viewMatrix * vec4(1.0, 0.0, 0.0, 0.0)).xyz);
    vec3 pnormal = normalize((_280.viewMatrix * light.lightDirection).xyz);
    vec3 ppos = (_280.viewMatrix * light.lightPosition).xyz;
    vec3 up = normalize(cross(pnormal, right));
    right = normalize(cross(up, pnormal));
    float width = light.lightSize.x;
    float height = light.lightSize.y;
    vec3 param = _input.v.xyz;
    vec3 param_1 = ppos;
    vec3 param_2 = pnormal;
    vec3 projection = projectOnPlane(param, param_1, param_2);
    vec3 dir = projection - ppos;
    vec2 diagonal = vec2(dot(dir, right), dot(dir, up));
    vec2 nearest2D = vec2(clamp(diagonal.x, -width, width), clamp(diagonal.y, -height, height));
    vec3 nearestPointInside = (ppos + (right * nearest2D.x)) + (up * nearest2D.y);
    vec3 L = nearestPointInside - _input.v.xyz;
    float lightToSurfDist = length(L);
    L = normalize(L);
    float param_3 = lightToSurfDist;
    int param_4 = light.lightDistAttenCurveType;
    vec4 param_5[2] = light.lightDistAttenCurveParams;
    float atten = apply_atten_curve(param_3, param_4, param_5);
    vec3 linearColor = vec3(0.0);
    float pnDotL = dot(pnormal, -L);
    float nDotL = dot(N, L);
    vec3 param_6 = _input.v.xyz;
    vec3 param_7 = ppos;
    vec3 param_8 = pnormal;
    if ((nDotL > 0.0) && isAbovePlane(param_6, param_7, param_8))
    {
        vec3 V = normalize(-_input.v.xyz);
        vec3 R = normalize((N * (2.0 * dot(V, N))) - V);
        vec3 R2 = normalize((N * (2.0 * dot(L, N))) - L);
        vec3 param_9 = _input.v.xyz;
        vec3 param_10 = R;
        vec3 param_11 = ppos;
        vec3 param_12 = pnormal;
        vec3 E = linePlaneIntersect(param_9, param_10, param_11, param_12);
        float specAngle = clamp(dot(-R, pnormal), 0.0, 1.0);
        vec3 dirSpec = E - ppos;
        vec2 dirSpec2D = vec2(dot(dirSpec, right), dot(dirSpec, up));
        vec2 nearestSpec2D = vec2(clamp(dirSpec2D.x, -width, width), clamp(dirSpec2D.y, -height, height));
        float specFactor = 1.0 - clamp(length(nearestSpec2D - dirSpec2D), 0.0, 1.0);
        vec3 admit_light = light.lightColor.xyz * (light.lightIntensity * atten);
        linearColor = (texture(SPIRV_Cross_CombineddiffuseMapsamp0, _input.uv).xyz * nDotL) * pnDotL;
        linearColor += (((vec3(0.800000011920928955078125) * pow(clamp(dot(R2, V), 0.0, 1.0), 50.0)) * specFactor) * specAngle);
        linearColor *= admit_light;
    }
    return linearColor;
}

vec3 apply_light(Light light, basic_vert_output _input)
{
    vec3 N = normalize(_input.normal.xyz);
    vec3 light_dir = normalize((_280.viewMatrix * light.lightDirection).xyz);
    vec3 L;
    if (light.lightPosition.w == 0.0)
    {
        L = -light_dir;
    }
    else
    {
        L = (_280.viewMatrix * light.lightPosition).xyz - _input.v.xyz;
    }
    float lightToSurfDist = length(L);
    L = normalize(L);
    float cosTheta = clamp(dot(N, L), 0.0, 1.0);
    float visibility = 1.0;
    float lightToSurfAngle = acos(dot(L, -light_dir));
    float param = lightToSurfAngle;
    int param_1 = light.lightAngleAttenCurveType;
    vec4 param_2[2] = light.lightAngleAttenCurveParams;
    float atten = apply_atten_curve(param, param_1, param_2);
    float param_3 = lightToSurfDist;
    int param_4 = light.lightDistAttenCurveType;
    vec4 param_5[2] = light.lightDistAttenCurveParams;
    atten *= apply_atten_curve(param_3, param_4, param_5);
    vec3 R = normalize((N * (2.0 * dot(L, N))) - L);
    vec3 V = normalize(-_input.v.xyz);
    vec3 admit_light = light.lightColor.xyz * (light.lightIntensity * atten);
    vec3 linearColor = texture(SPIRV_Cross_CombineddiffuseMapsamp0, _input.uv).xyz * cosTheta;
    if (visibility > 0.20000000298023223876953125)
    {
        linearColor += vec3(0.800000011920928955078125 * pow(clamp(dot(R, V), 0.0, 1.0), 50.0));
    }
    linearColor *= admit_light;
    return linearColor * visibility;
}

vec3 exposure_tone_mapping(vec3 color)
{
    return vec3(1.0) - exp((-color) * 1.0);
}

vec4 _basic_frag_main(basic_vert_output _entryPointOutput_1)
{
    vec3 linearColor = vec3(0.0);
    for (int i = 0; i < _280.numLights; i++)
    {
        if (_677.lights[i].lightType == 3)
        {
            Light arg;
            arg.lightIntensity = _677.lights[i].lightIntensity;
            arg.lightType = _677.lights[i].lightType;
            arg.lightCastShadow = _677.lights[i].lightCastShadow;
            arg.lightShadowMapIndex = _677.lights[i].lightShadowMapIndex;
            arg.lightAngleAttenCurveType = _677.lights[i].lightAngleAttenCurveType;
            arg.lightDistAttenCurveType = _677.lights[i].lightDistAttenCurveType;
            arg.lightSize = _677.lights[i].lightSize;
            arg.lightGuid = _677.lights[i].lightGuid;
            arg.lightPosition = _677.lights[i].lightPosition;
            arg.lightColor = _677.lights[i].lightColor;
            arg.lightDirection = _677.lights[i].lightDirection;
            arg.lightDistAttenCurveParams[0] = _677.lights[i].lightDistAttenCurveParams[0];
            arg.lightDistAttenCurveParams[1] = _677.lights[i].lightDistAttenCurveParams[1];
            arg.lightAngleAttenCurveParams[0] = _677.lights[i].lightAngleAttenCurveParams[0];
            arg.lightAngleAttenCurveParams[1] = _677.lights[i].lightAngleAttenCurveParams[1];
            arg.lightVP = _677.lights[i].lightVP;
            arg.padding[0] = _677.lights[i].padding[0];
            arg.padding[1] = _677.lights[i].padding[1];
            basic_vert_output param = _entryPointOutput_1;
            linearColor += apply_areaLight(arg, param);
        }
        else
        {
            Light arg_1;
            arg_1.lightIntensity = _677.lights[i].lightIntensity;
            arg_1.lightType = _677.lights[i].lightType;
            arg_1.lightCastShadow = _677.lights[i].lightCastShadow;
            arg_1.lightShadowMapIndex = _677.lights[i].lightShadowMapIndex;
            arg_1.lightAngleAttenCurveType = _677.lights[i].lightAngleAttenCurveType;
            arg_1.lightDistAttenCurveType = _677.lights[i].lightDistAttenCurveType;
            arg_1.lightSize = _677.lights[i].lightSize;
            arg_1.lightGuid = _677.lights[i].lightGuid;
            arg_1.lightPosition = _677.lights[i].lightPosition;
            arg_1.lightColor = _677.lights[i].lightColor;
            arg_1.lightDirection = _677.lights[i].lightDirection;
            arg_1.lightDistAttenCurveParams[0] = _677.lights[i].lightDistAttenCurveParams[0];
            arg_1.lightDistAttenCurveParams[1] = _677.lights[i].lightDistAttenCurveParams[1];
            arg_1.lightAngleAttenCurveParams[0] = _677.lights[i].lightAngleAttenCurveParams[0];
            arg_1.lightAngleAttenCurveParams[1] = _677.lights[i].lightAngleAttenCurveParams[1];
            arg_1.lightVP = _677.lights[i].lightVP;
            arg_1.padding[0] = _677.lights[i].padding[0];
            arg_1.padding[1] = _677.lights[i].padding[1];
            basic_vert_output param_1 = _entryPointOutput_1;
            linearColor += apply_light(arg_1, param_1);
        }
    }
    linearColor += (textureLod(SPIRV_Cross_Combinedskyboxsamp0, vec4(_entryPointOutput_1.normal_world.xyz, 0.0), 2.0).xyz * vec3(0.20000000298023223876953125));
    vec3 param_2 = linearColor;
    linearColor = exposure_tone_mapping(param_2);
    return vec4(linearColor, 1.0);
}

void main()
{
    basic_vert_output _entryPointOutput_1;
    _entryPointOutput_1.pos = gl_FragCoord;
    _entryPointOutput_1.normal = _entryPointOutput_normal;
    _entryPointOutput_1.normal_world = _entryPointOutput_normal_world;
    _entryPointOutput_1.v = _entryPointOutput_v;
    _entryPointOutput_1.v_world = _entryPointOutput_v_world;
    _entryPointOutput_1.uv = _entryPointOutput_uv;
    basic_vert_output param = _entryPointOutput_1;
    _entryPointOutput = _basic_frag_main(param);
}

