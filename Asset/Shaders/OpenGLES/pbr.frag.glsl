#version 310 es
precision mediump float;
precision highp int;

struct vert_output
{
    highp vec4 position;
    highp vec4 normal;
    highp vec4 normal_world;
    highp vec4 v;
    highp vec4 v_world;
    highp vec2 uv;
    highp mat3 TBN;
    highp vec3 v_tangent;
    highp vec3 camPos_tangent;
};

struct Light
{
    highp float lightIntensity;
    uint lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    uint lightAngleAttenCurveType;
    uint lightDistAttenCurveType;
    highp vec2 lightSize;
    uvec4 lightGuid;
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
    layout(row_major) highp mat4 viewMatrix;
    layout(row_major) highp mat4 projectionMatrix;
    highp vec4 camPos;
    uint numLights;
    highp float padding[3];
    Light lights[100];
} _765;

uniform highp sampler2D SPIRV_Cross_CombinedheightMapsamp0;
uniform highp sampler2D SPIRV_Cross_CombinednormalMapsamp0;
uniform highp sampler2D SPIRV_Cross_CombineddiffuseMapsamp0;
uniform highp sampler2D SPIRV_Cross_CombinedmetalicMapsamp0;
uniform highp sampler2D SPIRV_Cross_CombinedroughnessMapsamp0;
uniform highp samplerCubeArray SPIRV_Cross_CombinedcubeShadowMapsamp0;
uniform highp sampler2DArray SPIRV_Cross_CombinedshadowMapsamp0;
uniform highp sampler2DArray SPIRV_Cross_CombinedglobalShadowMapsamp0;
uniform highp sampler2D SPIRV_Cross_CombinedaoMapsamp0;
uniform highp samplerCubeArray SPIRV_Cross_Combinedskyboxsamp0;
uniform highp sampler2D SPIRV_Cross_CombinedbrdfLUTsamp0;

layout(location = 0) in highp vec4 input_normal;
layout(location = 1) in highp vec4 input_normal_world;
layout(location = 2) in highp vec4 input_v;
layout(location = 3) in highp vec4 input_v_world;
layout(location = 4) in highp vec2 input_uv;
layout(location = 5) in highp mat3 input_TBN;
layout(location = 8) in highp vec3 input_v_tangent;
layout(location = 9) in highp vec3 input_camPos_tangent;
layout(location = 0) out highp vec4 _entryPointOutput;

float _118;

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

highp vec2 ParallaxMapping(highp vec2 texCoords, highp vec3 viewDir)
{
    highp float param = abs(dot(vec3(0.0, 0.0, 1.0), viewDir));
    highp float param_1 = 32.0;
    highp float param_2 = 8.0;
    highp float numLayers = linear_interpolate(param, param_1, param_2);
    highp float layerDepth = 1.0 / numLayers;
    highp float currentLayerDepth = 0.0;
    highp vec2 currentTexCoords = texCoords;
    highp float currentDepthMapValue = 1.0 - texture(SPIRV_Cross_CombinedheightMapsamp0, currentTexCoords).x;
    highp vec2 P = viewDir.xy * 0.100000001490116119384765625;
    highp vec2 deltaTexCoords = P / vec2(numLayers);
    while (currentLayerDepth < currentDepthMapValue)
    {
        currentTexCoords -= deltaTexCoords;
        currentDepthMapValue = 1.0 - texture(SPIRV_Cross_CombinedheightMapsamp0, currentTexCoords).x;
        currentLayerDepth += layerDepth;
    }
    highp vec2 prevTexCoords = currentTexCoords + deltaTexCoords;
    highp float afterDepth = currentDepthMapValue - currentLayerDepth;
    highp float beforeDepth = ((1.0 - texture(SPIRV_Cross_CombinedheightMapsamp0, prevTexCoords).x) - currentLayerDepth) + layerDepth;
    highp float weight = afterDepth / (afterDepth - beforeDepth);
    highp vec2 finalTexCoords = (prevTexCoords * weight) + (currentTexCoords * (1.0 - weight));
    return finalTexCoords;
}

highp vec3 inverse_gamma_correction(highp vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(2.2000000476837158203125));
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
        int i;
        switch (light.lightType)
        {
            case 0:
            {
                highp vec3 L = p.xyz - light.lightPosition.xyz;
                near_occ = texture(SPIRV_Cross_CombinedcubeShadowMapsamp0, vec4(L, float(light.lightShadowMapIndex))).x;
                if ((length(L) - (near_occ * 10.0)) > bias)
                {
                    visibility -= 0.87999999523162841796875;
                }
                break;
            }
            case 1:
            {
                v_light_space *= mat4(vec4(0.5, 0.0, 0.0, 0.0), vec4(0.0, 0.5, 0.0, 0.0), vec4(0.0, 0.0, 0.5, 0.0), vec4(0.5, 0.5, 0.5, 1.0));
                i = 0;
                for (; i < 4; i++)
                {
                    highp mat4x2 indexable = mat4x2(vec2(-0.94201624393463134765625, -0.39906215667724609375), vec2(0.94558608531951904296875, -0.768907248973846435546875), vec2(-0.094184100627899169921875, -0.929388701915740966796875), vec2(0.34495937824249267578125, 0.29387760162353515625));
                    near_occ = texture(SPIRV_Cross_CombinedshadowMapsamp0, vec3(v_light_space.xy + (indexable[i] / vec2(700.0)), float(light.lightShadowMapIndex))).x;
                    if ((v_light_space.z - near_occ) > bias)
                    {
                        visibility -= 0.2199999988079071044921875;
                    }
                }
                break;
            }
            case 2:
            {
                v_light_space *= mat4(vec4(0.5, 0.0, 0.0, 0.0), vec4(0.0, 0.5, 0.0, 0.0), vec4(0.0, 0.0, 0.5, 0.0), vec4(0.5, 0.5, 0.5, 1.0));
                i = 0;
                for (; i < 4; i++)
                {
                    highp mat4x2 indexable_1 = mat4x2(vec2(-0.94201624393463134765625, -0.39906215667724609375), vec2(0.94558608531951904296875, -0.768907248973846435546875), vec2(-0.094184100627899169921875, -0.929388701915740966796875), vec2(0.34495937824249267578125, 0.29387760162353515625));
                    near_occ = texture(SPIRV_Cross_CombinedglobalShadowMapsamp0, vec3(v_light_space.xy + (indexable_1[i] / vec2(700.0)), float(light.lightShadowMapIndex))).x;
                    if ((v_light_space.z - near_occ) > bias)
                    {
                        visibility -= 0.2199999988079071044921875;
                    }
                }
                break;
            }
            case 3:
            {
                v_light_space *= mat4(vec4(0.5, 0.0, 0.0, 0.0), vec4(0.0, 0.5, 0.0, 0.0), vec4(0.0, 0.0, 0.5, 0.0), vec4(0.5, 0.5, 0.5, 1.0));
                i = 0;
                for (; i < 4; i++)
                {
                    highp mat4x2 indexable_2 = mat4x2(vec2(-0.94201624393463134765625, -0.39906215667724609375), vec2(0.94558608531951904296875, -0.768907248973846435546875), vec2(-0.094184100627899169921875, -0.929388701915740966796875), vec2(0.34495937824249267578125, 0.29387760162353515625));
                    near_occ = texture(SPIRV_Cross_CombinedshadowMapsamp0, vec3(v_light_space.xy + (indexable_2[i] / vec2(700.0)), float(light.lightShadowMapIndex))).x;
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
            highp float param_3 = param;
            highp float param_4 = param_1;
            highp float param_5 = param_2;
            atten = linear_interpolate(param_3, param_4, param_5);
            break;
        }
        case 2:
        {
            highp float begin_atten_1 = atten_params[0].x;
            highp float end_atten_1 = atten_params[0].y;
            highp float param_3_1 = dist;
            highp float param_4_1 = begin_atten_1;
            highp float param_5_1 = end_atten_1;
            highp float param_6 = param_3_1;
            highp float param_7 = param_4_1;
            highp float param_8 = param_5_1;
            highp float tmp = linear_interpolate(param_6, param_7, param_8);
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

highp float DistributionGGX(highp vec3 N, highp vec3 H, highp float roughness)
{
    highp float a = roughness * roughness;
    highp float a2 = a * a;
    highp float NdotH = max(dot(N, H), 0.0);
    highp float NdotH2 = NdotH * NdotH;
    highp float num = a2;
    highp float denom = (NdotH2 * (a2 - 1.0)) + 1.0;
    denom = (3.1415927410125732421875 * denom) * denom;
    return num / denom;
}

highp float GeometrySchlickGGXDirect(highp float NdotV, highp float roughness)
{
    highp float r = roughness + 1.0;
    highp float k = (r * r) / 8.0;
    highp float num = NdotV;
    highp float denom = (NdotV * (1.0 - k)) + k;
    return num / denom;
}

highp float GeometrySmithDirect(highp vec3 N, highp vec3 V, highp vec3 L, highp float roughness)
{
    highp float NdotV = max(dot(N, V), 0.0);
    highp float NdotL = max(dot(N, L), 0.0);
    highp float param = NdotV;
    highp float param_1 = roughness;
    highp float ggx2 = GeometrySchlickGGXDirect(param, param_1);
    highp float param_2 = NdotL;
    highp float param_3 = roughness;
    highp float ggx1 = GeometrySchlickGGXDirect(param_2, param_3);
    return ggx1 * ggx2;
}

highp vec3 fresnelSchlick(highp float cosTheta, highp vec3 F0)
{
    return F0 + ((vec3(1.0) - F0) * pow(1.0 - cosTheta, 5.0));
}

highp vec3 fresnelSchlickRoughness(highp float cosTheta, highp vec3 F0, highp float roughness)
{
    return F0 + ((max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0));
}

highp vec3 reinhard_tone_mapping(highp vec3 color)
{
    return color / (color + vec3(1.0));
}

highp vec3 gamma_correction(highp vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(0.4545454680919647216796875));
}

highp vec4 _pbr_frag_main(vert_output _input)
{
    highp vec3 viewDir = normalize(_input.camPos_tangent - _input.v_tangent);
    highp vec2 param = _input.uv;
    highp vec3 param_1 = viewDir;
    highp vec2 texCoords = ParallaxMapping(param, param_1);
    highp vec3 tangent_normal = texture(SPIRV_Cross_CombinednormalMapsamp0, texCoords).xyz;
    tangent_normal = (tangent_normal * 2.0) - vec3(1.0);
    highp vec3 N = normalize(_input.TBN * tangent_normal);
    highp vec3 V = normalize(_765.camPos.xyz - _input.v_world.xyz);
    highp vec3 R = reflect(-V, N);
    highp vec3 param_2 = texture(SPIRV_Cross_CombineddiffuseMapsamp0, texCoords).xyz;
    highp vec3 albedo = inverse_gamma_correction(param_2);
    highp float meta = texture(SPIRV_Cross_CombinedmetalicMapsamp0, texCoords).x;
    highp float rough = texture(SPIRV_Cross_CombinedroughnessMapsamp0, texCoords).x;
    highp vec3 F0 = vec3(0.039999999105930328369140625);
    F0 = mix(F0, albedo, vec3(meta));
    highp vec3 Lo = vec3(0.0);
    for (int i = 0; uint(i) < _765.numLights; i++)
    {
        Light light;
        light.lightIntensity = _765.lights[i].lightIntensity;
        light.lightType = _765.lights[i].lightType;
        light.lightCastShadow = _765.lights[i].lightCastShadow;
        light.lightShadowMapIndex = _765.lights[i].lightShadowMapIndex;
        light.lightAngleAttenCurveType = _765.lights[i].lightAngleAttenCurveType;
        light.lightDistAttenCurveType = _765.lights[i].lightDistAttenCurveType;
        light.lightSize = _765.lights[i].lightSize;
        light.lightGuid = _765.lights[i].lightGuid;
        light.lightPosition = _765.lights[i].lightPosition;
        light.lightColor = _765.lights[i].lightColor;
        light.lightDirection = _765.lights[i].lightDirection;
        light.lightDistAttenCurveParams[0] = _765.lights[i].lightDistAttenCurveParams[0];
        light.lightDistAttenCurveParams[1] = _765.lights[i].lightDistAttenCurveParams[1];
        light.lightAngleAttenCurveParams[0] = _765.lights[i].lightAngleAttenCurveParams[0];
        light.lightAngleAttenCurveParams[1] = _765.lights[i].lightAngleAttenCurveParams[1];
        light.lightVP = _765.lights[i].lightVP;
        light.padding[0] = _765.lights[i].padding[0];
        light.padding[1] = _765.lights[i].padding[1];
        highp vec3 L = normalize(light.lightPosition.xyz - _input.v_world.xyz);
        highp vec3 H = normalize(V + L);
        highp float NdotL = max(dot(N, L), 0.0);
        highp vec4 param_3 = _input.v_world;
        Light param_4 = light;
        highp float param_5 = NdotL;
        highp float visibility = shadow_test(param_3, param_4, param_5);
        highp float lightToSurfDist = length(L);
        highp float lightToSurfAngle = acos(dot(-L, light.lightDirection.xyz));
        highp float param_6 = lightToSurfAngle;
        int param_7 = int(light.lightAngleAttenCurveType);
        highp vec4 param_8[2] = light.lightAngleAttenCurveParams;
        highp float atten = apply_atten_curve(param_6, param_7, param_8);
        highp float param_9 = lightToSurfDist;
        int param_10 = int(light.lightDistAttenCurveType);
        highp vec4 param_11[2] = light.lightDistAttenCurveParams;
        atten *= apply_atten_curve(param_9, param_10, param_11);
        highp vec3 radiance = light.lightColor.xyz * (light.lightIntensity * atten);
        highp vec3 param_12 = N;
        highp vec3 param_13 = H;
        highp float param_14 = rough;
        highp float NDF = DistributionGGX(param_12, param_13, param_14);
        highp vec3 param_15 = N;
        highp vec3 param_16 = V;
        highp vec3 param_17 = L;
        highp float param_18 = rough;
        highp float G = GeometrySmithDirect(param_15, param_16, param_17, param_18);
        highp float param_19 = max(dot(H, V), 0.0);
        highp vec3 param_20 = F0;
        highp vec3 F = fresnelSchlick(param_19, param_20);
        highp vec3 kS = F;
        highp vec3 kD = vec3(1.0) - kS;
        kD *= (1.0 - meta);
        highp vec3 numerator = F * (NDF * G);
        highp float denominator = (4.0 * max(dot(N, V), 0.0)) * NdotL;
        highp vec3 specular = numerator / vec3(max(denominator, 0.001000000047497451305389404296875));
        Lo += ((((((kD * albedo) / vec3(3.1415927410125732421875)) + specular) * radiance) * NdotL) * visibility);
    }
    highp float ambientOcc = texture(SPIRV_Cross_CombinedaoMapsamp0, texCoords).x;
    highp float param_21 = max(dot(N, V), 0.0);
    highp vec3 param_22 = F0;
    highp float param_23 = rough;
    highp vec3 F_1 = fresnelSchlickRoughness(param_21, param_22, param_23);
    highp vec3 kS_1 = F_1;
    highp vec3 kD_1 = vec3(1.0) - kS_1;
    kD_1 *= (1.0 - meta);
    highp vec3 irradiance = textureLod(SPIRV_Cross_Combinedskyboxsamp0, vec4(N, 0.0), 1.0).xyz;
    highp vec3 diffuse = irradiance * albedo;
    highp vec3 prefilteredColor = textureLod(SPIRV_Cross_Combinedskyboxsamp0, vec4(R, 1.0), rough * 9.0).xyz;
    highp vec2 envBRDF = texture(SPIRV_Cross_CombinedbrdfLUTsamp0, vec2(max(dot(N, V), 0.0), rough)).xy;
    highp vec3 specular_1 = prefilteredColor * ((F_1 * envBRDF.x) + vec3(envBRDF.y));
    highp vec3 ambient = ((kD_1 * diffuse) + specular_1) * ambientOcc;
    highp vec3 linearColor = ambient + Lo;
    highp vec3 param_24 = linearColor;
    linearColor = reinhard_tone_mapping(param_24);
    highp vec3 param_25 = linearColor;
    linearColor = gamma_correction(param_25);
    return vec4(linearColor, 1.0);
}

void main()
{
    vert_output _input;
    _input.position = gl_FragCoord;
    _input.normal = input_normal;
    _input.normal_world = input_normal_world;
    _input.v = input_v;
    _input.v_world = input_v_world;
    _input.uv = input_uv;
    _input.TBN = input_TBN;
    _input.v_tangent = input_v_tangent;
    _input.camPos_tangent = input_camPos_tangent;
    vert_output param = _input;
    _entryPointOutput = _pbr_frag_main(param);
}

