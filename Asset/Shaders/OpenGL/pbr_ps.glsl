#version 400

struct Light
{
    float lightIntensity;
    int lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    vec2 lightSize;
    ivec4 lightGUID;
    vec4 lightPosition;
    vec4 lightColor;
    vec4 lightDirection;
    vec4 lightDistAttenCurveParams[2];
    vec4 lightAngleAttenCurveParams[2];
    mat4 lightVP;
    vec4 padding[2];
};

layout(std140) uniform PerFrameConstants
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 camPos;
    int numLights;
    Light allLights[100];
} _580;

uniform samplerCubeArray cubeShadowMap;
uniform sampler2DArray shadowMap;
uniform sampler2DArray globalShadowMap;
uniform sampler2D diffuseMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;
uniform samplerCubeArray skybox;
uniform sampler2D brdfLUT;

in vec4 normal_world;
in vec4 v_world;
in vec2 uv;
layout(location = 0) out vec4 outputColor;

float _100;

float shadow_test(vec4 p, Light light, float cosTheta)
{
    vec4 v_light_space = light.lightVP * p;
    v_light_space /= vec4(v_light_space.w);
    float visibility = 1.0;
    if (light.lightShadowMapIndex != (-1))
    {
        float bias = 0.0005000000237487256526947021484375 * tan(acos(cosTheta));
        bias = clamp(bias, 0.0, 0.00999999977648258209228515625);
        float near_occ;
        switch (light.lightType)
        {
            case 0:
            {
                vec3 L = p.xyz - light.lightPosition.xyz;
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
                    vec2 indexable[4] = vec2[](vec2(-0.94201624393463134765625, -0.39906215667724609375), vec2(0.94558608531951904296875, -0.768907248973846435546875), vec2(-0.094184100627899169921875, -0.929388701915740966796875), vec2(0.34495937824249267578125, 0.29387760162353515625));
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
                    vec2 indexable_1[4] = vec2[](vec2(-0.94201624393463134765625, -0.39906215667724609375), vec2(0.94558608531951904296875, -0.768907248973846435546875), vec2(-0.094184100627899169921875, -0.929388701915740966796875), vec2(0.34495937824249267578125, 0.29387760162353515625));
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
                    vec2 indexable_2[4] = vec2[](vec2(-0.94201624393463134765625, -0.39906215667724609375), vec2(0.94558608531951904296875, -0.768907248973846435546875), vec2(-0.094184100627899169921875, -0.929388701915740966796875), vec2(0.34495937824249267578125, 0.29387760162353515625));
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
            atten = linear_interpolate(param, param_1, param_2);
            break;
        }
        case 2:
        {
            float begin_atten_1 = atten_params[0].x;
            float end_atten_1 = atten_params[0].y;
            float param_3 = dist;
            float param_4 = begin_atten_1;
            float param_5 = end_atten_1;
            float tmp = linear_interpolate(param_3, param_4, param_5);
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

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0)) + 1.0;
    denom = (3.1415927410125732421875 * denom) * denom;
    return num / denom;
}

float GeometrySchlickGGXDirect(float NdotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float num = NdotV;
    float denom = (NdotV * (1.0 - k)) + k;
    return num / denom;
}

float GeometrySmithDirect(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float param = NdotV;
    float param_1 = roughness;
    float ggx2 = GeometrySchlickGGXDirect(param, param_1);
    float param_2 = NdotL;
    float param_3 = roughness;
    float ggx1 = GeometrySchlickGGXDirect(param_2, param_3);
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + ((vec3(1.0) - F0) * pow(1.0 - cosTheta, 5.0));
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + ((max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0));
}

vec3 reinhard_tone_mapping(vec3 color)
{
    return color / (color + vec3(1.0));
}

vec3 gamma_correction(vec3 color)
{
    return pow(color, vec3(0.4545454680919647216796875));
}

void main()
{
    vec3 N = normalize(normal_world.xyz);
    vec3 V = normalize(_580.camPos.xyz - v_world.xyz);
    vec3 R = reflect(-V, N);
    vec3 albedo = texture(diffuseMap, uv).xyz;
    float meta = texture(metallicMap, uv).x;
    float rough = texture(roughnessMap, uv).x;
    vec3 F0 = vec3(0.039999999105930328369140625);
    F0 = mix(F0, albedo, vec3(meta));
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < _580.numLights; i++)
    {
        Light light;
        light.lightIntensity = _580.allLights[i].lightIntensity;
        light.lightType = _580.allLights[i].lightType;
        light.lightCastShadow = _580.allLights[i].lightCastShadow;
        light.lightShadowMapIndex = _580.allLights[i].lightShadowMapIndex;
        light.lightAngleAttenCurveType = _580.allLights[i].lightAngleAttenCurveType;
        light.lightDistAttenCurveType = _580.allLights[i].lightDistAttenCurveType;
        light.lightSize = _580.allLights[i].lightSize;
        light.lightGUID = _580.allLights[i].lightGUID;
        light.lightPosition = _580.allLights[i].lightPosition;
        light.lightColor = _580.allLights[i].lightColor;
        light.lightDirection = _580.allLights[i].lightDirection;
        light.lightDistAttenCurveParams[0] = _580.allLights[i].lightDistAttenCurveParams[0];
        light.lightDistAttenCurveParams[1] = _580.allLights[i].lightDistAttenCurveParams[1];
        light.lightAngleAttenCurveParams[0] = _580.allLights[i].lightAngleAttenCurveParams[0];
        light.lightAngleAttenCurveParams[1] = _580.allLights[i].lightAngleAttenCurveParams[1];
        light.lightVP = _580.allLights[i].lightVP;
        light.padding[0] = _580.allLights[i].padding[0];
        light.padding[1] = _580.allLights[i].padding[1];
        vec3 L = normalize(light.lightPosition.xyz - v_world.xyz);
        vec3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);
        float visibility = shadow_test(v_world, light, NdotL);
        float lightToSurfDist = length(L);
        float lightToSurfAngle = acos(dot(-L, light.lightDirection.xyz));
        float param = lightToSurfAngle;
        int param_1 = light.lightAngleAttenCurveType;
        vec4 param_2[2] = light.lightAngleAttenCurveParams;
        float atten = apply_atten_curve(param, param_1, param_2);
        float param_3 = lightToSurfDist;
        int param_4 = light.lightDistAttenCurveType;
        vec4 param_5[2] = light.lightDistAttenCurveParams;
        atten *= apply_atten_curve(param_3, param_4, param_5);
        vec3 radiance = light.lightColor.xyz * (light.lightIntensity * atten);
        vec3 param_6 = N;
        vec3 param_7 = H;
        float param_8 = rough;
        float NDF = DistributionGGX(param_6, param_7, param_8);
        vec3 param_9 = N;
        vec3 param_10 = V;
        vec3 param_11 = L;
        float param_12 = rough;
        float G = GeometrySmithDirect(param_9, param_10, param_11, param_12);
        float param_13 = max(dot(H, V), 0.0);
        vec3 param_14 = F0;
        vec3 F = fresnelSchlick(param_13, param_14);
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= (1.0 - meta);
        vec3 numerator = F * (NDF * G);
        float denominator = (4.0 * max(dot(N, V), 0.0)) * NdotL;
        vec3 specular = numerator / vec3(max(denominator, 0.001000000047497451305389404296875));
        Lo += ((((((kD * albedo) / vec3(3.1415927410125732421875)) + specular) * radiance) * NdotL) * visibility);
    }
    float ambientOcc = texture(aoMap, uv).x;
    float param_15 = max(dot(N, V), 0.0);
    vec3 param_16 = F0;
    float param_17 = rough;
    vec3 F_1 = fresnelSchlickRoughness(param_15, param_16, param_17);
    vec3 kS_1 = F_1;
    vec3 kD_1 = vec3(1.0) - kS_1;
    kD_1 *= (1.0 - meta);
    vec3 irradiance = textureLod(skybox, vec4(N, 0.0), 1.0).xyz;
    vec3 diffuse = irradiance * albedo;
    vec3 prefilteredColor = textureLod(skybox, vec4(R, 1.0), rough * 8.0).xyz;
    vec2 envBRDF = texture(brdfLUT, vec2(max(dot(N, V), 0.0), rough)).xy;
    vec3 specular_1 = prefilteredColor * ((F_1 * envBRDF.x) + vec3(envBRDF.y));
    vec3 ambient = ((kD_1 * diffuse) + specular_1) * ambientOcc;
    vec3 linearColor = ambient + Lo;
    vec3 param_18 = linearColor;
    linearColor = reinhard_tone_mapping(param_18);
    vec3 param_19 = linearColor;
    linearColor = gamma_correction(param_19);
    outputColor = vec4(linearColor, 1.0);
}

