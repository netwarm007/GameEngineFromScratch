#version 400

struct pbr_vert_output
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
} _411;

layout(std140) uniform LightInfo
{
    Light lights[100];
} _489;

uniform sampler2D SPIRV_Cross_CombinednormalMapsamp0;
uniform sampler2D SPIRV_Cross_CombineddiffuseMapsamp0;
uniform sampler2D SPIRV_Cross_CombinedmetallicMapsamp0;
uniform sampler2D SPIRV_Cross_CombinedroughnessMapsamp0;
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

float _101;

vec3 inverse_gamma_correction(vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(2.2000000476837158203125));
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

vec4 _pbr_frag_main(pbr_vert_output _input)
{
    vec3 viewDir = normalize(_input.camPos_tangent - _input.v_tangent);
    vec2 texCoords = _input.uv;
    vec3 tangent_normal = texture(SPIRV_Cross_CombinednormalMapsamp0, texCoords).xyz;
    tangent_normal = (tangent_normal * 2.0) - vec3(1.0);
    vec3 N = normalize(_input.TBN * tangent_normal);
    vec3 V = normalize(_411.camPos.xyz - _input.v_world.xyz);
    vec3 R = reflect(-V, N);
    vec3 param = texture(SPIRV_Cross_CombineddiffuseMapsamp0, texCoords).xyz;
    vec3 albedo = inverse_gamma_correction(param);
    float meta = texture(SPIRV_Cross_CombinedmetallicMapsamp0, texCoords).x;
    float rough = texture(SPIRV_Cross_CombinedroughnessMapsamp0, texCoords).x;
    vec3 F0 = vec3(0.039999999105930328369140625);
    F0 = mix(F0, albedo, vec3(meta));
    vec3 Lo = vec3(0.0);
    for (int i = 0; uint(i) < _411.numLights; i++)
    {
        Light light;
        light.lightIntensity = _489.lights[i].lightIntensity;
        light.lightType = _489.lights[i].lightType;
        light.lightCastShadow = _489.lights[i].lightCastShadow;
        light.lightShadowMapIndex = _489.lights[i].lightShadowMapIndex;
        light.lightAngleAttenCurveType = _489.lights[i].lightAngleAttenCurveType;
        light.lightDistAttenCurveType = _489.lights[i].lightDistAttenCurveType;
        light.lightSize = _489.lights[i].lightSize;
        light.lightGuid = _489.lights[i].lightGuid;
        light.lightPosition = _489.lights[i].lightPosition;
        light.lightColor = _489.lights[i].lightColor;
        light.lightDirection = _489.lights[i].lightDirection;
        light.lightDistAttenCurveParams[0] = _489.lights[i].lightDistAttenCurveParams[0];
        light.lightDistAttenCurveParams[1] = _489.lights[i].lightDistAttenCurveParams[1];
        light.lightAngleAttenCurveParams[0] = _489.lights[i].lightAngleAttenCurveParams[0];
        light.lightAngleAttenCurveParams[1] = _489.lights[i].lightAngleAttenCurveParams[1];
        light.lightVP = _489.lights[i].lightVP;
        light.padding[0] = _489.lights[i].padding[0];
        light.padding[1] = _489.lights[i].padding[1];
        vec3 L = normalize(light.lightPosition.xyz - _input.v_world.xyz);
        vec3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);
        float visibility = 1.0;
        float lightToSurfDist = length(L);
        float lightToSurfAngle = acos(dot(-L, light.lightDirection.xyz));
        float param_1 = lightToSurfAngle;
        int param_2 = int(light.lightAngleAttenCurveType);
        vec4 param_3[2] = light.lightAngleAttenCurveParams;
        float atten = apply_atten_curve(param_1, param_2, param_3);
        float param_4 = lightToSurfDist;
        int param_5 = int(light.lightDistAttenCurveType);
        vec4 param_6[2] = light.lightDistAttenCurveParams;
        atten *= apply_atten_curve(param_4, param_5, param_6);
        vec3 radiance = light.lightColor.xyz * (light.lightIntensity * atten);
        vec3 param_7 = N;
        vec3 param_8 = H;
        float param_9 = rough;
        float NDF = DistributionGGX(param_7, param_8, param_9);
        vec3 param_10 = N;
        vec3 param_11 = V;
        vec3 param_12 = L;
        float param_13 = rough;
        float G = GeometrySmithDirect(param_10, param_11, param_12, param_13);
        float param_14 = max(dot(H, V), 0.0);
        vec3 param_15 = F0;
        vec3 F = fresnelSchlick(param_14, param_15);
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= (1.0 - meta);
        vec3 numerator = F * (NDF * G);
        float denominator = (4.0 * max(dot(N, V), 0.0)) * NdotL;
        vec3 specular = numerator / vec3(max(denominator, 0.001000000047497451305389404296875));
        Lo += ((((((kD * albedo) / vec3(3.1415927410125732421875)) + specular) * radiance) * NdotL) * visibility);
    }
    float ambientOcc = texture(SPIRV_Cross_CombinedaoMapsamp0, texCoords).x;
    float param_16 = max(dot(N, V), 0.0);
    vec3 param_17 = F0;
    float param_18 = rough;
    vec3 F_1 = fresnelSchlickRoughness(param_16, param_17, param_18);
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
    vec3 param_19 = linearColor;
    linearColor = reinhard_tone_mapping(param_19);
    vec3 param_20 = linearColor;
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
    _input.uv = input_uv;
    _input.TBN = input_TBN;
    _input.v_tangent = input_v_tangent;
    _input.camPos_tangent = input_camPos_tangent;
    pbr_vert_output param = _input;
    _entryPointOutput = _pbr_frag_main(param);
}

