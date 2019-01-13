#version 420

struct pbr_vert_output
{
    vec4 pos;
    vec4 normal;
    vec4 normal_world;
    vec4 v;
    vec4 v_world;
    vec3 v_tangent;
    vec3 camPos_tangent;
    vec2 uv;
    mat3 TBN;
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

layout(binding = 10, std140) uniform PerFrameConstants
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    mat4 arbitraryMatrix;
    vec4 camPos;
    uint numLights;
} _596;

layout(binding = 12, std140) uniform LightInfo
{
    Light lights[100];
} _671;

uniform sampler2D SPIRV_Cross_CombinednormalMapsamp0;
uniform sampler2D SPIRV_Cross_CombineddiffuseMapsamp0;
uniform sampler2D SPIRV_Cross_CombinedmetallicMapsamp0;
uniform sampler2D SPIRV_Cross_CombinedroughnessMapsamp0;
uniform sampler2D SPIRV_Cross_CombinedaoMapsamp0;
uniform sampler2DArray SPIRV_Cross_Combinedskyboxsamp0;
uniform sampler2D SPIRV_Cross_CombinedbrdfLUTsamp0;

layout(location = 0) in vec4 _entryPointOutput_normal;
layout(location = 1) in vec4 _entryPointOutput_normal_world;
layout(location = 2) in vec4 _entryPointOutput_v;
layout(location = 3) in vec4 _entryPointOutput_v_world;
layout(location = 4) in vec3 _entryPointOutput_v_tangent;
layout(location = 5) in vec3 _entryPointOutput_camPos_tangent;
layout(location = 6) in vec2 _entryPointOutput_uv;
layout(location = 7) in mat3 _entryPointOutput_TBN;
layout(location = 0) out vec4 _entryPointOutput;

float _104;

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

vec3 convert_xyz_to_cube_uv(vec3 d)
{
    vec3 d_abs = abs(d);
    bvec3 isPositive;
    isPositive.x = int(d.x > 0.0) != int(0u);
    isPositive.y = int(d.y > 0.0) != int(0u);
    isPositive.z = int(d.z > 0.0) != int(0u);
    float maxAxis;
    float uc;
    float vc;
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
    vec3 o;
    o.x = 0.5 * ((uc / maxAxis) + 1.0);
    o.y = 0.5 * ((vc / maxAxis) + 1.0);
    o.z = float(index);
    return o;
}

vec3 reinhard_tone_mapping(vec3 color)
{
    return color / (color + vec3(1.0));
}

vec3 gamma_correction(vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(0.4545454680919647216796875));
}

vec4 _pbr_frag_main(pbr_vert_output _entryPointOutput_1)
{
    vec2 texCoords = _entryPointOutput_1.uv;
    vec3 tangent_normal = texture(SPIRV_Cross_CombinednormalMapsamp0, texCoords).xyz;
    tangent_normal = (tangent_normal * 2.0) - vec3(1.0);
    vec3 N = normalize(_entryPointOutput_1.TBN * tangent_normal);
    vec3 V = normalize(_596.camPos.xyz - _entryPointOutput_1.v_world.xyz);
    vec3 R = reflect(-V, N);
    vec3 param = texture(SPIRV_Cross_CombineddiffuseMapsamp0, texCoords).xyz;
    vec3 albedo = inverse_gamma_correction(param);
    float meta = texture(SPIRV_Cross_CombinedmetallicMapsamp0, texCoords).x;
    float rough = texture(SPIRV_Cross_CombinedroughnessMapsamp0, texCoords).x;
    vec3 F0 = vec3(0.039999999105930328369140625);
    F0 = mix(F0, albedo, vec3(meta));
    vec3 Lo = vec3(0.0);
    for (int i = 0; uint(i) < _596.numLights; i++)
    {
        Light light;
        light.lightIntensity = _671.lights[i].lightIntensity;
        light.lightType = _671.lights[i].lightType;
        light.lightCastShadow = _671.lights[i].lightCastShadow;
        light.lightShadowMapIndex = _671.lights[i].lightShadowMapIndex;
        light.lightAngleAttenCurveType = _671.lights[i].lightAngleAttenCurveType;
        light.lightDistAttenCurveType = _671.lights[i].lightDistAttenCurveType;
        light.lightSize = _671.lights[i].lightSize;
        light.lightGuid = _671.lights[i].lightGuid;
        light.lightPosition = _671.lights[i].lightPosition;
        light.lightColor = _671.lights[i].lightColor;
        light.lightDirection = _671.lights[i].lightDirection;
        light.lightDistAttenCurveParams[0] = _671.lights[i].lightDistAttenCurveParams[0];
        light.lightDistAttenCurveParams[1] = _671.lights[i].lightDistAttenCurveParams[1];
        light.lightAngleAttenCurveParams[0] = _671.lights[i].lightAngleAttenCurveParams[0];
        light.lightAngleAttenCurveParams[1] = _671.lights[i].lightAngleAttenCurveParams[1];
        light.lightVP = _671.lights[i].lightVP;
        light.padding[0] = _671.lights[i].padding[0];
        light.padding[1] = _671.lights[i].padding[1];
        vec3 L = normalize(light.lightPosition.xyz - _entryPointOutput_1.v_world.xyz);
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
    vec3 param_19 = N;
    vec3 uvw = convert_xyz_to_cube_uv(param_19);
    vec3 irradiance = textureLod(SPIRV_Cross_Combinedskyboxsamp0, N, 1.0).xyz;
    vec3 diffuse = irradiance * albedo;
    vec3 param_20 = R;
    vec3 uvw_1 = convert_xyz_to_cube_uv(param_20);
    vec3 prefilteredColor = textureLod(SPIRV_Cross_Combinedskyboxsamp0, uvw_1, rough * 9.0).xyz;
    vec2 envBRDF = texture(SPIRV_Cross_CombinedbrdfLUTsamp0, vec2(max(dot(N, V), 0.0), rough)).xy;
    vec3 specular_1 = prefilteredColor * ((F_1 * envBRDF.x) + vec3(envBRDF.y));
    vec3 ambient = ((kD_1 * diffuse) + specular_1) * ambientOcc;
    vec3 linearColor = ambient + Lo;
    vec3 param_21 = linearColor;
    linearColor = reinhard_tone_mapping(param_21);
    vec3 param_22 = linearColor;
    linearColor = gamma_correction(param_22);
    return vec4(linearColor, 1.0);
}

void main()
{
    pbr_vert_output _entryPointOutput_1;
    _entryPointOutput_1.pos = gl_FragCoord;
    _entryPointOutput_1.normal = _entryPointOutput_normal;
    _entryPointOutput_1.normal_world = _entryPointOutput_normal_world;
    _entryPointOutput_1.v = _entryPointOutput_v;
    _entryPointOutput_1.v_world = _entryPointOutput_v_world;
    _entryPointOutput_1.v_tangent = _entryPointOutput_v_tangent;
    _entryPointOutput_1.camPos_tangent = _entryPointOutput_camPos_tangent;
    _entryPointOutput_1.uv = _entryPointOutput_uv;
    _entryPointOutput_1.TBN = _entryPointOutput_TBN;
    pbr_vert_output param = _entryPointOutput_1;
    _entryPointOutput = _pbr_frag_main(param);
}

