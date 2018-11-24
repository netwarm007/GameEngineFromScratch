#version 400

struct vert_output
{
    vec4 pos;
    vec4 normal;
    vec4 normal_world;
    vec4 v;
    vec4 v_world;
    vec2 uv;
    mat3 TBN;
    vec3 v_tangent;
    vec3 camPos_tangent;
};

struct Light
{
    float lightIntensity;
    uint lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    uint lightAngleAttenCurveType;
    uint lightDistAttenCurveType;
    vec2 lightSize;
    uvec4 lightGuid;
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
    uint numLights;
} _758;

layout(std140) uniform LightInfo
{
    Light lights[100];
} _830;

uniform sampler2D SPIRV_Cross_CombinedheightMapsamp0;
uniform sampler2D SPIRV_Cross_CombinednormalMapsamp0;
uniform sampler2D SPIRV_Cross_CombineddiffuseMapsamp0;
uniform sampler2D SPIRV_Cross_CombinedmetallicMapsamp0;
uniform sampler2D SPIRV_Cross_CombinedroughnessMapsamp0;
uniform samplerCubeArray SPIRV_Cross_CombinedcubeShadowMapsamp0;
uniform sampler2DArray SPIRV_Cross_CombinedshadowMapsamp0;
uniform sampler2DArray SPIRV_Cross_CombinedglobalShadowMapsamp0;
uniform sampler2D SPIRV_Cross_CombinedaoMapsamp0;
uniform samplerCubeArray SPIRV_Cross_Combinedskyboxsamp0;
uniform sampler2D SPIRV_Cross_CombinedbrdfLUTsamp0;

in vec4 input_normal;
in vec4 input_normal_world;
in vec4 input_v;
in vec4 input_v_world;
in vec2 input_uv;
in mat3 input_TBN;
in vec3 input_v_tangent;
in vec3 input_camPos_tangent;
layout(location = 0) out vec4 _entryPointOutput;

float _118;

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

vec2 ParallaxMapping(vec2 texCoords, vec3 viewDir)
{
    float param = abs(dot(vec3(0.0, 0.0, 1.0), viewDir));
    float param_1 = 32.0;
    float param_2 = 8.0;
    float numLayers = linear_interpolate(param, param_1, param_2);
    float layerDepth = 1.0 / numLayers;
    float currentLayerDepth = 0.0;
    vec2 currentTexCoords = texCoords;
    float currentDepthMapValue = 1.0 - texture(SPIRV_Cross_CombinedheightMapsamp0, currentTexCoords).x;
    vec2 P = viewDir.xy * 0.100000001490116119384765625;
    vec2 deltaTexCoords = P / vec2(numLayers);
    while (currentLayerDepth < currentDepthMapValue)
    {
        currentTexCoords -= deltaTexCoords;
        currentDepthMapValue = 1.0 - texture(SPIRV_Cross_CombinedheightMapsamp0, currentTexCoords).x;
        currentLayerDepth += layerDepth;
    }
    vec2 prevTexCoords = currentTexCoords + deltaTexCoords;
    float afterDepth = currentDepthMapValue - currentLayerDepth;
    float beforeDepth = ((1.0 - texture(SPIRV_Cross_CombinedheightMapsamp0, prevTexCoords).x) - currentLayerDepth) + layerDepth;
    float weight = afterDepth / (afterDepth - beforeDepth);
    vec2 finalTexCoords = (prevTexCoords * weight) + (currentTexCoords * (1.0 - weight));
    return finalTexCoords;
}

vec3 inverse_gamma_correction(vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(2.2000000476837158203125));
}

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
        int i;
        switch (light.lightType)
        {
            case 0:
            {
                vec3 L = p.xyz - light.lightPosition.xyz;
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
                    mat4x2 indexable = mat4x2(vec2(-0.94201624393463134765625, -0.39906215667724609375), vec2(0.94558608531951904296875, -0.768907248973846435546875), vec2(-0.094184100627899169921875, -0.929388701915740966796875), vec2(0.34495937824249267578125, 0.29387760162353515625));
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
                    mat4x2 indexable_1 = mat4x2(vec2(-0.94201624393463134765625, -0.39906215667724609375), vec2(0.94558608531951904296875, -0.768907248973846435546875), vec2(-0.094184100627899169921875, -0.929388701915740966796875), vec2(0.34495937824249267578125, 0.29387760162353515625));
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
                    mat4x2 indexable_2 = mat4x2(vec2(-0.94201624393463134765625, -0.39906215667724609375), vec2(0.94558608531951904296875, -0.768907248973846435546875), vec2(-0.094184100627899169921875, -0.929388701915740966796875), vec2(0.34495937824249267578125, 0.29387760162353515625));
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
    return pow(max(color, vec3(0.0)), vec3(0.4545454680919647216796875));
}

vec4 _pbr_frag_main(vert_output _input)
{
    vec3 viewDir = normalize(_input.camPos_tangent - _input.v_tangent);
    vec2 param = _input.uv;
    vec3 param_1 = viewDir;
    vec2 texCoords = ParallaxMapping(param, param_1);
    vec3 tangent_normal = texture(SPIRV_Cross_CombinednormalMapsamp0, texCoords).xyz;
    tangent_normal = (tangent_normal * 2.0) - vec3(1.0);
    vec3 N = normalize(_input.TBN * tangent_normal);
    vec3 V = normalize(_758.camPos.xyz - _input.v_world.xyz);
    vec3 R = reflect(-V, N);
    vec3 param_2 = texture(SPIRV_Cross_CombineddiffuseMapsamp0, texCoords).xyz;
    vec3 albedo = inverse_gamma_correction(param_2);
    float meta = texture(SPIRV_Cross_CombinedmetallicMapsamp0, texCoords).x;
    float rough = texture(SPIRV_Cross_CombinedroughnessMapsamp0, texCoords).x;
    vec3 F0 = vec3(0.039999999105930328369140625);
    F0 = mix(F0, albedo, vec3(meta));
    vec3 Lo = vec3(0.0);
    for (int i = 0; uint(i) < _758.numLights; i++)
    {
        Light light;
        light.lightIntensity = _830.lights[i].lightIntensity;
        light.lightType = _830.lights[i].lightType;
        light.lightCastShadow = _830.lights[i].lightCastShadow;
        light.lightShadowMapIndex = _830.lights[i].lightShadowMapIndex;
        light.lightAngleAttenCurveType = _830.lights[i].lightAngleAttenCurveType;
        light.lightDistAttenCurveType = _830.lights[i].lightDistAttenCurveType;
        light.lightSize = _830.lights[i].lightSize;
        light.lightGuid = _830.lights[i].lightGuid;
        light.lightPosition = _830.lights[i].lightPosition;
        light.lightColor = _830.lights[i].lightColor;
        light.lightDirection = _830.lights[i].lightDirection;
        light.lightDistAttenCurveParams[0] = _830.lights[i].lightDistAttenCurveParams[0];
        light.lightDistAttenCurveParams[1] = _830.lights[i].lightDistAttenCurveParams[1];
        light.lightAngleAttenCurveParams[0] = _830.lights[i].lightAngleAttenCurveParams[0];
        light.lightAngleAttenCurveParams[1] = _830.lights[i].lightAngleAttenCurveParams[1];
        light.lightVP = _830.lights[i].lightVP;
        light.padding[0] = _830.lights[i].padding[0];
        light.padding[1] = _830.lights[i].padding[1];
        vec3 L = normalize(light.lightPosition.xyz - _input.v_world.xyz);
        vec3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);
        vec4 param_3 = _input.v_world;
        Light param_4 = light;
        float param_5 = NdotL;
        float visibility = shadow_test(param_3, param_4, param_5);
        float lightToSurfDist = length(L);
        float lightToSurfAngle = acos(dot(-L, light.lightDirection.xyz));
        float param_6 = lightToSurfAngle;
        int param_7 = int(light.lightAngleAttenCurveType);
        vec4 param_8[2] = light.lightAngleAttenCurveParams;
        float atten = apply_atten_curve(param_6, param_7, param_8);
        float param_9 = lightToSurfDist;
        int param_10 = int(light.lightDistAttenCurveType);
        vec4 param_11[2] = light.lightDistAttenCurveParams;
        atten *= apply_atten_curve(param_9, param_10, param_11);
        vec3 radiance = light.lightColor.xyz * (light.lightIntensity * atten);
        vec3 param_12 = N;
        vec3 param_13 = H;
        float param_14 = rough;
        float NDF = DistributionGGX(param_12, param_13, param_14);
        vec3 param_15 = N;
        vec3 param_16 = V;
        vec3 param_17 = L;
        float param_18 = rough;
        float G = GeometrySmithDirect(param_15, param_16, param_17, param_18);
        float param_19 = max(dot(H, V), 0.0);
        vec3 param_20 = F0;
        vec3 F = fresnelSchlick(param_19, param_20);
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= (1.0 - meta);
        vec3 numerator = F * (NDF * G);
        float denominator = (4.0 * max(dot(N, V), 0.0)) * NdotL;
        vec3 specular = numerator / vec3(max(denominator, 0.001000000047497451305389404296875));
        Lo += ((((((kD * albedo) / vec3(3.1415927410125732421875)) + specular) * radiance) * NdotL) * visibility);
    }
    float ambientOcc = texture(SPIRV_Cross_CombinedaoMapsamp0, texCoords).x;
    float param_21 = max(dot(N, V), 0.0);
    vec3 param_22 = F0;
    float param_23 = rough;
    vec3 F_1 = fresnelSchlickRoughness(param_21, param_22, param_23);
    vec3 kS_1 = F_1;
    vec3 kD_1 = vec3(1.0) - kS_1;
    kD_1 *= (1.0 - meta);
    vec3 irradiance = textureLod(SPIRV_Cross_Combinedskyboxsamp0, vec4(N, 0.0), 1.0).xyz;
    vec3 diffuse = irradiance * albedo;
    vec3 prefilteredColor = textureLod(SPIRV_Cross_Combinedskyboxsamp0, vec4(R, 1.0), rough * 9.0).xyz;
    vec2 envBRDF = texture(SPIRV_Cross_CombinedbrdfLUTsamp0, vec2(max(dot(N, V), 0.0), rough)).xy;
    vec3 specular_1 = prefilteredColor * ((F_1 * envBRDF.x) + vec3(envBRDF.y));
    vec3 ambient = ((kD_1 * diffuse) + specular_1) * ambientOcc;
    vec3 linearColor = ambient + Lo;
    vec3 param_24 = linearColor;
    linearColor = reinhard_tone_mapping(param_24);
    vec3 param_25 = linearColor;
    linearColor = gamma_correction(param_25);
    return vec4(linearColor, 1.0);
}

void main()
{
    vert_output _input;
    _input.pos = gl_FragCoord;
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

