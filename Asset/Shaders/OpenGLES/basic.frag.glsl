#version 310 es
precision mediump float;
precision highp int;

struct Light
{
    highp float lightIntensity;
    int lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    highp vec2 lightSize;
    ivec4 lightGUID;
    highp vec4 lightPosition;
    highp vec4 lightColor;
    highp vec4 lightDirection;
    highp vec4 lightDistAttenCurveParams[2];
    highp vec4 lightAngleAttenCurveParams[2];
    highp mat4 lightVP;
    highp vec4 padding[2];
};

layout(binding = 0, std140) uniform PerFrameConstants
{
    highp mat4 viewMatrix;
    highp mat4 projectionMatrix;
    highp vec4 camPos;
    int numLights;
    Light allLights[100];
} _89;

struct constants_t
{
    highp vec4 ambientColor;
    highp vec4 specularColor;
    highp float specularPower;
};

uniform constants_t u_pushConstants;

layout(binding = 0) uniform highp sampler2D diffuseMap;
layout(binding = 4) uniform highp samplerCubeArray skybox;
layout(binding = 3) uniform highp samplerCubeArray cubeShadowMap;
layout(binding = 1) uniform highp sampler2DArray shadowMap;
layout(binding = 2) uniform highp sampler2DArray globalShadowMap;

layout(location = 0) in highp vec4 normal;
layout(location = 2) in highp vec4 v;
layout(location = 3) in highp vec4 v_world;
layout(location = 4) in highp vec2 uv;
layout(location = 1) in highp vec4 normal_world;
layout(location = 0) out highp vec4 outputColor;

float _728;

highp vec3 projectOnPlane(highp vec3 point, highp vec3 center_of_plane, highp vec3 normal_of_plane)
{
    return point - (normal_of_plane * dot(point - center_of_plane, normal_of_plane));
}

highp float linear_interpolate(highp float t, highp float begin, highp float end)
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

highp float apply_atten_curve(highp float dist, int atten_curve_type, highp vec4 atten_params[2])
{
    highp float atten = 1.0;
    switch (atten_curve_type)
    {
        case 1:
        {
            highp float begin_atten = atten_params[0].x;
            highp float end_atten = atten_params[0].y;
            highp float param = dist;
            highp float param_1 = begin_atten;
            highp float param_2 = end_atten;
            atten = linear_interpolate(param, param_1, param_2);
            break;
        }
        case 2:
        {
            highp float begin_atten_1 = atten_params[0].x;
            highp float end_atten_1 = atten_params[0].y;
            highp float param_3 = dist;
            highp float param_4 = begin_atten_1;
            highp float param_5 = end_atten_1;
            highp float tmp = linear_interpolate(param_3, param_4, param_5);
            atten = (3.0 * pow(tmp, 2.0)) - (2.0 * pow(tmp, 3.0));
            break;
        }
        case 3:
        {
            highp float scale = atten_params[0].x;
            highp float offset = atten_params[0].y;
            highp float kl = atten_params[0].z;
            highp float kc = atten_params[0].w;
            atten = clamp((scale / ((kl * dist) + (kc * scale))) + offset, 0.0, 1.0);
            break;
        }
        case 4:
        {
            highp float scale_1 = atten_params[0].x;
            highp float offset_1 = atten_params[0].y;
            highp float kq = atten_params[0].z;
            highp float kl_1 = atten_params[0].w;
            highp float kc_1 = atten_params[1].x;
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

bool isAbovePlane(highp vec3 point, highp vec3 center_of_plane, highp vec3 normal_of_plane)
{
    return dot(point - center_of_plane, normal_of_plane) > 0.0;
}

highp vec3 linePlaneIntersect(highp vec3 line_start, highp vec3 line_dir, highp vec3 center_of_plane, highp vec3 normal_of_plane)
{
    return line_start + (line_dir * (dot(center_of_plane - line_start, normal_of_plane) / dot(line_dir, normal_of_plane)));
}

highp vec3 apply_areaLight(Light light)
{
    highp vec3 N = normalize(normal.xyz);
    highp vec3 right = normalize((_89.viewMatrix * vec4(1.0, 0.0, 0.0, 0.0)).xyz);
    highp vec3 pnormal = normalize((_89.viewMatrix * light.lightDirection).xyz);
    highp vec3 ppos = (_89.viewMatrix * light.lightPosition).xyz;
    highp vec3 up = normalize(cross(pnormal, right));
    right = normalize(cross(up, pnormal));
    highp float width = light.lightSize.x;
    highp float height = light.lightSize.y;
    highp vec3 param = v.xyz;
    highp vec3 param_1 = ppos;
    highp vec3 param_2 = pnormal;
    highp vec3 projection = projectOnPlane(param, param_1, param_2);
    highp vec3 dir = projection - ppos;
    highp vec2 diagonal = vec2(dot(dir, right), dot(dir, up));
    highp vec2 nearest2D = vec2(clamp(diagonal.x, -width, width), clamp(diagonal.y, -height, height));
    highp vec3 nearestPointInside = (ppos + (right * nearest2D.x)) + (up * nearest2D.y);
    highp vec3 L = nearestPointInside - v.xyz;
    highp float lightToSurfDist = length(L);
    L = normalize(L);
    highp float param_3 = lightToSurfDist;
    int param_4 = light.lightDistAttenCurveType;
    highp vec4 param_5[2] = light.lightDistAttenCurveParams;
    highp float atten = apply_atten_curve(param_3, param_4, param_5);
    highp vec3 linearColor = vec3(0.0);
    highp float pnDotL = dot(pnormal, -L);
    highp float nDotL = dot(N, L);
    highp float _389 = nDotL;
    bool _390 = _389 > 0.0;
    bool _401;
    if (_390)
    {
        highp vec3 param_6 = v.xyz;
        highp vec3 param_7 = ppos;
        highp vec3 param_8 = pnormal;
        _401 = isAbovePlane(param_6, param_7, param_8);
    }
    else
    {
        _401 = _390;
    }
    if (_401)
    {
        highp vec3 V = normalize(-v.xyz);
        highp vec3 R = normalize((N * (2.0 * dot(V, N))) - V);
        highp vec3 R2 = normalize((N * (2.0 * dot(L, N))) - L);
        highp vec3 param_9 = v.xyz;
        highp vec3 param_10 = R;
        highp vec3 param_11 = ppos;
        highp vec3 param_12 = pnormal;
        highp vec3 E = linePlaneIntersect(param_9, param_10, param_11, param_12);
        highp float specAngle = clamp(dot(-R, pnormal), 0.0, 1.0);
        highp vec3 dirSpec = E - ppos;
        highp vec2 dirSpec2D = vec2(dot(dirSpec, right), dot(dirSpec, up));
        highp vec2 nearestSpec2D = vec2(clamp(dirSpec2D.x, -width, width), clamp(dirSpec2D.y, -height, height));
        highp float specFactor = 1.0 - clamp(length(nearestSpec2D - dirSpec2D), 0.0, 1.0);
        highp vec3 admit_light = light.lightColor.xyz * (light.lightIntensity * atten);
        linearColor = (texture(diffuseMap, uv).xyz * nDotL) * pnDotL;
        linearColor += (((u_pushConstants.specularColor.xyz * pow(clamp(dot(R2, V), 0.0, 1.0), u_pushConstants.specularPower)) * specFactor) * specAngle);
        linearColor *= admit_light;
    }
    return linearColor;
}

highp float shadow_test(highp vec4 p, Light light, highp float cosTheta)
{
    highp vec4 v_light_space = light.lightVP * p;
    v_light_space /= vec4(v_light_space.w);
    highp float visibility = 1.0;
    if (light.lightShadowMapIndex != (-1))
    {
        highp float bias = 0.0005000000237487256526947021484375 * tan(acos(cosTheta));
        bias = clamp(bias, 0.0, 0.00999999977648258209228515625);
        highp float near_occ;
        switch (light.lightType)
        {
            case 0:
            {
                highp vec3 L = p.xyz - light.lightPosition.xyz;
                near_occ = texture(cubeShadowMap, vec4(L, float(light.lightShadowMapIndex))).x;
                if ((length(L) - (near_occ * 10.0)) > bias)
                {
                    visibility -= 0.87999999523162841796875;
                }
                break;
            }
            case 1:
            {
                v_light_space = mat4(vec4(0.5, 0.0, 0.0, 0.0), vec4(0.0, 0.5, 0.0, 0.0), vec4(0.0, 0.0, 0.5, 0.0), vec4(0.5, 0.5, 0.5, 1.0)) * v_light_space;
                for (int i = 0; i < 4; i++)
                {
                    highp vec2 indexable[4] = vec2[](vec2(-0.94201624393463134765625, -0.39906215667724609375), vec2(0.94558608531951904296875, -0.768907248973846435546875), vec2(-0.094184100627899169921875, -0.929388701915740966796875), vec2(0.34495937824249267578125, 0.29387760162353515625));
                    near_occ = texture(shadowMap, vec3(v_light_space.xy + (indexable[i] / vec2(700.0)), float(light.lightShadowMapIndex))).x;
                    if ((v_light_space.z - near_occ) > bias)
                    {
                        visibility -= 0.2199999988079071044921875;
                    }
                }
                break;
            }
            case 2:
            {
                v_light_space = mat4(vec4(0.5, 0.0, 0.0, 0.0), vec4(0.0, 0.5, 0.0, 0.0), vec4(0.0, 0.0, 0.5, 0.0), vec4(0.5, 0.5, 0.5, 1.0)) * v_light_space;
                for (int i_1 = 0; i_1 < 4; i_1++)
                {
                    highp vec2 indexable_1[4] = vec2[](vec2(-0.94201624393463134765625, -0.39906215667724609375), vec2(0.94558608531951904296875, -0.768907248973846435546875), vec2(-0.094184100627899169921875, -0.929388701915740966796875), vec2(0.34495937824249267578125, 0.29387760162353515625));
                    near_occ = texture(globalShadowMap, vec3(v_light_space.xy + (indexable_1[i_1] / vec2(700.0)), float(light.lightShadowMapIndex))).x;
                    if ((v_light_space.z - near_occ) > bias)
                    {
                        visibility -= 0.2199999988079071044921875;
                    }
                }
                break;
            }
            case 3:
            {
                v_light_space = mat4(vec4(0.5, 0.0, 0.0, 0.0), vec4(0.0, 0.5, 0.0, 0.0), vec4(0.0, 0.0, 0.5, 0.0), vec4(0.5, 0.5, 0.5, 1.0)) * v_light_space;
                for (int i_2 = 0; i_2 < 4; i_2++)
                {
                    highp vec2 indexable_2[4] = vec2[](vec2(-0.94201624393463134765625, -0.39906215667724609375), vec2(0.94558608531951904296875, -0.768907248973846435546875), vec2(-0.094184100627899169921875, -0.929388701915740966796875), vec2(0.34495937824249267578125, 0.29387760162353515625));
                    near_occ = texture(shadowMap, vec3(v_light_space.xy + (indexable_2[i_2] / vec2(700.0)), float(light.lightShadowMapIndex))).x;
                    if ((v_light_space.z - near_occ) > bias)
                    {
                        visibility -= 0.2199999988079071044921875;
                    }
                }
                break;
            }
        }
    }
    return visibility;
}

highp vec3 apply_light(Light light)
{
    highp vec3 N = normalize(normal.xyz);
    highp vec3 light_dir = normalize((_89.viewMatrix * light.lightDirection).xyz);
    highp vec3 L;
    if (light.lightPosition.w == 0.0)
    {
        L = -light_dir;
    }
    else
    {
        L = (_89.viewMatrix * light.lightPosition).xyz - v.xyz;
    }
    highp float lightToSurfDist = length(L);
    L = normalize(L);
    highp float cosTheta = clamp(dot(N, L), 0.0, 1.0);
    Light arg;
    arg.lightIntensity = light.lightIntensity;
    arg.lightType = light.lightType;
    arg.lightCastShadow = light.lightCastShadow;
    arg.lightShadowMapIndex = light.lightShadowMapIndex;
    arg.lightAngleAttenCurveType = light.lightAngleAttenCurveType;
    arg.lightDistAttenCurveType = light.lightDistAttenCurveType;
    arg.lightSize = light.lightSize;
    arg.lightGUID = light.lightGUID;
    arg.lightPosition = light.lightPosition;
    arg.lightColor = light.lightColor;
    arg.lightDirection = light.lightDirection;
    arg.lightDistAttenCurveParams = light.lightDistAttenCurveParams;
    arg.lightAngleAttenCurveParams = light.lightAngleAttenCurveParams;
    arg.lightVP = light.lightVP;
    arg.padding = light.padding;
    highp float visibility = shadow_test(v_world, arg, cosTheta);
    highp float lightToSurfAngle = acos(dot(L, -light_dir));
    highp float param = lightToSurfAngle;
    int param_1 = light.lightAngleAttenCurveType;
    highp vec4 param_2[2] = light.lightAngleAttenCurveParams;
    highp float atten = apply_atten_curve(param, param_1, param_2);
    highp float param_3 = lightToSurfDist;
    int param_4 = light.lightDistAttenCurveType;
    highp vec4 param_5[2] = light.lightDistAttenCurveParams;
    atten *= apply_atten_curve(param_3, param_4, param_5);
    highp vec3 R = normalize((N * (2.0 * dot(L, N))) - L);
    highp vec3 V = normalize(-v.xyz);
    highp vec3 admit_light = light.lightColor.xyz * (light.lightIntensity * atten);
    highp vec3 linearColor = texture(diffuseMap, uv).xyz * cosTheta;
    if (visibility > 0.20000000298023223876953125)
    {
        linearColor += (u_pushConstants.specularColor.xyz * pow(clamp(dot(R, V), 0.0, 1.0), u_pushConstants.specularPower));
    }
    linearColor *= admit_light;
    return linearColor * visibility;
}

highp vec3 exposure_tone_mapping(highp vec3 color)
{
    return vec3(1.0) - exp((-color) * 1.0);
}

highp vec3 gamma_correction(highp vec3 color)
{
    return pow(color, vec3(0.4545454680919647216796875));
}

void main()
{
    highp vec3 linearColor = vec3(0.0);
    for (int i = 0; i < _89.numLights; i++)
    {
        if (_89.allLights[i].lightType == 3)
        {
            Light arg;
            arg.lightIntensity = _89.allLights[i].lightIntensity;
            arg.lightType = _89.allLights[i].lightType;
            arg.lightCastShadow = _89.allLights[i].lightCastShadow;
            arg.lightShadowMapIndex = _89.allLights[i].lightShadowMapIndex;
            arg.lightAngleAttenCurveType = _89.allLights[i].lightAngleAttenCurveType;
            arg.lightDistAttenCurveType = _89.allLights[i].lightDistAttenCurveType;
            arg.lightSize = _89.allLights[i].lightSize;
            arg.lightGUID = _89.allLights[i].lightGUID;
            arg.lightPosition = _89.allLights[i].lightPosition;
            arg.lightColor = _89.allLights[i].lightColor;
            arg.lightDirection = _89.allLights[i].lightDirection;
            arg.lightDistAttenCurveParams[0] = _89.allLights[i].lightDistAttenCurveParams[0];
            arg.lightDistAttenCurveParams[1] = _89.allLights[i].lightDistAttenCurveParams[1];
            arg.lightAngleAttenCurveParams[0] = _89.allLights[i].lightAngleAttenCurveParams[0];
            arg.lightAngleAttenCurveParams[1] = _89.allLights[i].lightAngleAttenCurveParams[1];
            arg.lightVP = _89.allLights[i].lightVP;
            arg.padding[0] = _89.allLights[i].padding[0];
            arg.padding[1] = _89.allLights[i].padding[1];
            linearColor += apply_areaLight(arg);
        }
        else
        {
            Light arg_1;
            arg_1.lightIntensity = _89.allLights[i].lightIntensity;
            arg_1.lightType = _89.allLights[i].lightType;
            arg_1.lightCastShadow = _89.allLights[i].lightCastShadow;
            arg_1.lightShadowMapIndex = _89.allLights[i].lightShadowMapIndex;
            arg_1.lightAngleAttenCurveType = _89.allLights[i].lightAngleAttenCurveType;
            arg_1.lightDistAttenCurveType = _89.allLights[i].lightDistAttenCurveType;
            arg_1.lightSize = _89.allLights[i].lightSize;
            arg_1.lightGUID = _89.allLights[i].lightGUID;
            arg_1.lightPosition = _89.allLights[i].lightPosition;
            arg_1.lightColor = _89.allLights[i].lightColor;
            arg_1.lightDirection = _89.allLights[i].lightDirection;
            arg_1.lightDistAttenCurveParams[0] = _89.allLights[i].lightDistAttenCurveParams[0];
            arg_1.lightDistAttenCurveParams[1] = _89.allLights[i].lightDistAttenCurveParams[1];
            arg_1.lightAngleAttenCurveParams[0] = _89.allLights[i].lightAngleAttenCurveParams[0];
            arg_1.lightAngleAttenCurveParams[1] = _89.allLights[i].lightAngleAttenCurveParams[1];
            arg_1.lightVP = _89.allLights[i].lightVP;
            arg_1.padding[0] = _89.allLights[i].padding[0];
            arg_1.padding[1] = _89.allLights[i].padding[1];
            linearColor += apply_light(arg_1);
        }
    }
    linearColor += (textureLod(skybox, vec4(normal_world.xyz, 0.0), 8.0).xyz * vec3(0.20000000298023223876953125));
    highp vec3 param = linearColor;
    linearColor = exposure_tone_mapping(param);
    highp vec3 param_1 = linearColor;
    outputColor = vec4(gamma_correction(param_1), 1.0);
}

