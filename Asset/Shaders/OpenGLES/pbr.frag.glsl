#version 310 es
precision mediump float;
precision highp int;

struct pbr_vert_output
{
    highp vec4 pos;
    highp vec4 normal;
    highp vec4 normal_world;
    highp vec4 v;
    highp vec4 v_world;
    highp vec3 v_tangent;
    highp vec3 camPos_tangent;
    highp vec2 uv;
    highp mat3 TBN;
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

layout(binding = 10, std140) uniform PerFrameConstants
{
    highp mat4 viewMatrix;
    highp mat4 projectionMatrix;
    highp mat4 arbitraryMatrix;
    highp vec4 camPos;
    uint numLights;
} _402;

layout(binding = 12, std140) uniform LightInfo
{
    Light lights[100];
} _479;

uniform highp sampler2D SPIRV_Cross_CombinednormalMapsamp0;
uniform highp sampler2D SPIRV_Cross_CombineddiffuseMapsamp0;
uniform highp sampler2D SPIRV_Cross_CombinedmetallicMapsamp0;
uniform highp sampler2D SPIRV_Cross_CombinedroughnessMapsamp0;
uniform highp sampler2D SPIRV_Cross_CombinedaoMapsamp0;
uniform highp samplerCubeArray SPIRV_Cross_Combinedskyboxsamp0;
uniform highp sampler2D SPIRV_Cross_CombinedbrdfLUTsamp0;

layout(location = 0) in highp vec4 input_normal;
layout(location = 1) in highp vec4 input_normal_world;
layout(location = 2) in highp vec4 input_v;
layout(location = 3) in highp vec4 input_v_world;
layout(location = 4) in highp vec3 input_v_tangent;
layout(location = 5) in highp vec3 input_camPos_tangent;
layout(location = 6) in highp vec2 input_uv;
layout(location = 7) in highp mat3 input_TBN;
layout(location = 0) out highp vec4 _entryPointOutput;

float _101;

highp vec3 inverse_gamma_correction(highp vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(2.2000000476837158203125));
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

highp vec4 _pbr_frag_main(pbr_vert_output _input)
{
    highp vec2 texCoords = _input.uv;
    highp vec3 tangent_normal = texture(SPIRV_Cross_CombinednormalMapsamp0, texCoords).xyz;
    tangent_normal = (tangent_normal * 2.0) - vec3(1.0);
    highp vec3 N = normalize(_input.TBN * tangent_normal);
    highp vec3 V = normalize(_402.camPos.xyz - _input.v_world.xyz);
    highp vec3 R = reflect(-V, N);
    highp vec3 param = texture(SPIRV_Cross_CombineddiffuseMapsamp0, texCoords).xyz;
    highp vec3 albedo = inverse_gamma_correction(param);
    highp float meta = texture(SPIRV_Cross_CombinedmetallicMapsamp0, texCoords).x;
    highp float rough = texture(SPIRV_Cross_CombinedroughnessMapsamp0, texCoords).x;
    highp vec3 F0 = vec3(0.039999999105930328369140625);
    F0 = mix(F0, albedo, vec3(meta));
    highp vec3 Lo = vec3(0.0);
    for (int i = 0; uint(i) < _402.numLights; i++)
    {
        Light light;
        light.lightIntensity = _479.lights[i].lightIntensity;
        light.lightType = _479.lights[i].lightType;
        light.lightCastShadow = _479.lights[i].lightCastShadow;
        light.lightShadowMapIndex = _479.lights[i].lightShadowMapIndex;
        light.lightAngleAttenCurveType = _479.lights[i].lightAngleAttenCurveType;
        light.lightDistAttenCurveType = _479.lights[i].lightDistAttenCurveType;
        light.lightSize = _479.lights[i].lightSize;
        light.lightGuid = _479.lights[i].lightGuid;
        light.lightPosition = _479.lights[i].lightPosition;
        light.lightColor = _479.lights[i].lightColor;
        light.lightDirection = _479.lights[i].lightDirection;
        light.lightDistAttenCurveParams[0] = _479.lights[i].lightDistAttenCurveParams[0];
        light.lightDistAttenCurveParams[1] = _479.lights[i].lightDistAttenCurveParams[1];
        light.lightAngleAttenCurveParams[0] = _479.lights[i].lightAngleAttenCurveParams[0];
        light.lightAngleAttenCurveParams[1] = _479.lights[i].lightAngleAttenCurveParams[1];
        light.lightVP = _479.lights[i].lightVP;
        light.padding[0] = _479.lights[i].padding[0];
        light.padding[1] = _479.lights[i].padding[1];
        highp vec3 L = normalize(light.lightPosition.xyz - _input.v_world.xyz);
        highp vec3 H = normalize(V + L);
        highp float NdotL = max(dot(N, L), 0.0);
        highp float visibility = 1.0;
        highp float lightToSurfDist = length(L);
        highp float lightToSurfAngle = acos(dot(-L, light.lightDirection.xyz));
        highp float param_1 = lightToSurfAngle;
        int param_2 = int(light.lightAngleAttenCurveType);
        highp vec4 param_3[2] = light.lightAngleAttenCurveParams;
        highp float atten = apply_atten_curve(param_1, param_2, param_3);
        highp float param_4 = lightToSurfDist;
        int param_5 = int(light.lightDistAttenCurveType);
        highp vec4 param_6[2] = light.lightDistAttenCurveParams;
        atten *= apply_atten_curve(param_4, param_5, param_6);
        highp vec3 radiance = light.lightColor.xyz * (light.lightIntensity * atten);
        highp vec3 param_7 = N;
        highp vec3 param_8 = H;
        highp float param_9 = rough;
        highp float NDF = DistributionGGX(param_7, param_8, param_9);
        highp vec3 param_10 = N;
        highp vec3 param_11 = V;
        highp vec3 param_12 = L;
        highp float param_13 = rough;
        highp float G = GeometrySmithDirect(param_10, param_11, param_12, param_13);
        highp float param_14 = max(dot(H, V), 0.0);
        highp vec3 param_15 = F0;
        highp vec3 F = fresnelSchlick(param_14, param_15);
        highp vec3 kS = F;
        highp vec3 kD = vec3(1.0) - kS;
        kD *= (1.0 - meta);
        highp vec3 numerator = F * (NDF * G);
        highp float denominator = (4.0 * max(dot(N, V), 0.0)) * NdotL;
        highp vec3 specular = numerator / vec3(max(denominator, 0.001000000047497451305389404296875));
        Lo += ((((((kD * albedo) / vec3(3.1415927410125732421875)) + specular) * radiance) * NdotL) * visibility);
    }
    highp float ambientOcc = texture(SPIRV_Cross_CombinedaoMapsamp0, texCoords).x;
    highp float param_16 = max(dot(N, V), 0.0);
    highp vec3 param_17 = F0;
    highp float param_18 = rough;
    highp vec3 F_1 = fresnelSchlickRoughness(param_16, param_17, param_18);
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
    highp vec3 param_19 = linearColor;
    linearColor = reinhard_tone_mapping(param_19);
    highp vec3 param_20 = linearColor;
    linearColor = gamma_correction(param_20);
    return vec4(linearColor, 1.0);
}

void main()
{
    pbr_vert_output _input;
    _input.pos = gl_FragCoord;
    _input.normal = input_normal;
    _input.normal_world = input_normal_world;
    _input.v = input_v;
    _input.v_world = input_v_world;
    _input.v_tangent = input_v_tangent;
    _input.camPos_tangent = input_camPos_tangent;
    _input.uv = input_uv;
    _input.TBN = input_TBN;
    pbr_vert_output param = _input;
    _entryPointOutput = _pbr_frag_main(param);
}

