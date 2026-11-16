cbuffer PerFrameConstants : register(b10)
{
    row_major float4x4 viewMatrix;
    row_major float4x4 projectionMatrix;
    float4 camPos;
    int numLights;
    int output_is_srgb;
};

cbuffer PerBatchConstants : register(b11)
{
    row_major float4x4 modelMatrix;
    float4 baseColor;
    float4 materialParameters;
};

struct Light {
    row_major float4x4 lightViewMatrix;
    row_major float4x4 lightProjectionMatrix;
    float lightIntensity;
    int lightType;
    int lightCastShadow;
    int lightShadowMapIndex;
    int lightAngleAttenCurveType;
    int lightDistAttenCurveType;
    float2 lightSize;
    int4 lightGuid;
    float4 lightPosition;
    float4 lightColor;
    float4 lightDirection;
    float4 lightDistAttenCurveParams[2];
    float4 lightAngleAttenCurveParams[2];
};

StructuredBuffer<Light> lights : register(t12);

Texture2D baseColorMap : register(t0);
Texture2D normalMap : register(t1);
Texture2D metallicMap : register(t2);
Texture2D roughnessMap : register(t3);
Texture2D aoMap : register(t4);
TextureCubeArray environmentMap : register(t6);
Texture2DArray shadowMaps : register(t7);
Texture2DArray globalShadowMaps : register(t8);
TextureCubeArray cubeShadowMaps : register(t9);
Texture2D brdfLUT : register(t14);
SamplerState materialSampler : register(s5);
static const float pointShadowFarPlane = 1000.0f;
static const float4x2 shadowPoissonDisk = float4x2(
    float2(-0.94201624f, -0.39906216f),
    float2(0.94558609f, -0.76890725f),
    float2(-0.094184101f, -0.92938870f),
    float2(0.34495938f, 0.29387760f));

float linear_interpolate(float value, float beginValue, float endValue)
{
    if (value < beginValue) return 1.0f;
    if (value > endValue) return 0.0f;
    return (endValue - value) / max(endValue - beginValue, 0.0001f);
}

float apply_atten_curve(float value, int curveType, float4 params[2])
{
    if (curveType == 1) {
        return linear_interpolate(value, params[0].x, params[0].y);
    }
    if (curveType == 2) {
        float factor = linear_interpolate(value, params[0].x, params[0].y);
        return 3.0f * factor * factor - 2.0f * factor * factor * factor;
    }
    if (curveType == 3) {
        return saturate(params[0].x /
                        (params[0].z * value + params[0].w * params[0].x) +
                        params[0].y);
    }
    if (curveType == 4) {
        float scale = params[0].x;
        float denominator = params[0].z * value * value +
                            params[0].w * value * scale +
                            params[1].x * scale * scale + params[0].y;
        return saturate(scale * scale / max(denominator, 0.0001f));
    }
    return 1.0f;
}

float3 fresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0f.xxx - F0) * pow(1.0f - cosTheta, 5.0f);
}

float distributionGGX(float3 normal, float3 halfway, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float nDotH = max(dot(normal, halfway), 0.0f);
    float denominator = nDotH * nDotH * (a2 - 1.0f) + 1.0f;
    return a2 / max(3.14159265f * denominator * denominator, 0.0001f);
}

float geometrySchlickGGX(float nDotV, float roughness)
{
    float r = roughness + 1.0f;
    float k = r * r / 8.0f;
    return nDotV / max(nDotV * (1.0f - k) + k, 0.0001f);
}

float geometrySmith(float3 normal, float3 viewDirection,
                    float3 lightDirection, float roughness)
{
    float nDotV = max(dot(normal, viewDirection), 0.0f);
    float nDotL = max(dot(normal, lightDirection), 0.0f);
    return geometrySchlickGGX(nDotV, roughness) *
           geometrySchlickGGX(nDotL, roughness);
}

float4 vulkan_scene_frag_main(
    float2 texCoords : TEXCOORD0,
    float3 worldPosition : TEXCOORD1,
    float3 worldNormal : TEXCOORD2,
    float3 worldTangent : TEXCOORD3) : SV_Target
{
    float4 albedoSample = baseColorMap.Sample(materialSampler, texCoords);
    float3 linearAlbedo =
        pow(max(albedoSample.rgb, 0.0f.xxx), 2.2f.xxx);
    float3 albedo = max(baseColor.rgb * linearAlbedo, 0.0f.xxx);
    float metallic = saturate(materialParameters.x *
                              metallicMap.Sample(materialSampler, texCoords).r);
    float roughness = clamp(
        materialParameters.y *
            roughnessMap.Sample(materialSampler, texCoords).r,
        0.04f, 1.0f);
    float ambientOcclusion = saturate(
        materialParameters.z * aoMap.Sample(materialSampler, texCoords).r);

    float3 N = normalize(worldNormal);
    float3 T = normalize(worldTangent - N * dot(N, worldTangent));
    float3 B = normalize(cross(N, T));
    float2 tangentNormalXY =
        normalMap.Sample(materialSampler, texCoords).rg * 2.0f - 1.0f;
    float3 tangentNormal = float3(
        tangentNormalXY,
        sqrt(saturate(1.0f - dot(tangentNormalXY, tangentNormalXY))));
    N = normalize(tangentNormal.x * T + tangentNormal.y * B +
                  tangentNormal.z * N);
    float3 V = normalize(camPos.xyz - worldPosition);
    float3 F0 = lerp(0.04f.xxx, albedo, metallic);
    float3 directLighting = 0.0f.xxx;

    for (int i = 0; i < numLights; ++i) {
        Light light = lights[i];
        float3 toLight = light.lightPosition.xyz - worldPosition;
        float distanceToLight = length(toLight);
        float3 L = light.lightPosition.w == 0.0f
                       ? normalize(-light.lightDirection.xyz)
                       : normalize(toLight);
        float attenuation =
            light.lightPosition.w == 0.0f
                ? 1.0f
                : apply_atten_curve(distanceToLight,
                                    light.lightDistAttenCurveType,
                                    light.lightDistAttenCurveParams);
        if (light.lightType == 1) {
            float lightAngle =
                acos(clamp(dot(-L, normalize(light.lightDirection.xyz)),
                           -1.0f, 1.0f));
            attenuation *= apply_atten_curve(
                lightAngle, light.lightAngleAttenCurveType,
                light.lightAngleAttenCurveParams);
        }

        float nDotL = max(dot(N, L), 0.0f);
        if (nDotL <= 0.0f) continue;

        float visibility = 1.0f;
        if (light.lightCastShadow != 0 && light.lightType == 0) {
            float3 lightToSurface =
                worldPosition - light.lightPosition.xyz;
            float storedDepth =
                cubeShadowMaps
                    .SampleLevel(
                        materialSampler,
                        float4(lightToSurface,
                               float(light.lightShadowMapIndex)),
                        0.0f)
                    .r;
            float currentDepth =
                length(lightToSurface) / pointShadowFarPlane;
            float pointBias =
                clamp(0.0005f * tan(acos(saturate(nDotL))), 0.0f, 0.01f) /
                pointShadowFarPlane;
            visibility =
                currentDepth - pointBias <= storedDepth ? 1.0f : 0.12f;
        } else if (light.lightCastShadow != 0 &&
                   (light.lightType == 1 || light.lightType == 2 ||
                    light.lightType == 3)) {
            float4 shadowClip =
                mul(mul(float4(worldPosition, 1.0f), light.lightViewMatrix),
                    light.lightProjectionMatrix);
            if (shadowClip.w > 0.0f) {
                float3 shadowCoordinates =
                    shadowClip.xyz / shadowClip.w;
                shadowCoordinates.xy =
                    shadowCoordinates.xy * 0.5f + 0.5f;
                if (all(shadowCoordinates.xy >= 0.0f.xx) &&
                    all(shadowCoordinates.xy <= 1.0f.xx) &&
                    shadowCoordinates.z >= 0.0f &&
                    shadowCoordinates.z <= 1.0f) {
                    float bias = clamp(
                        0.0005f * tan(acos(saturate(nDotL))), 0.0f, 0.01f);
                    uint shadowWidth;
                    uint shadowHeight;
                    uint shadowLayers;
                    uint shadowMipLevels;
                    float visibleSamples = 0.0f;
                    if (light.lightType == 2) {
                        globalShadowMaps.GetDimensions(
                            0, shadowWidth, shadowHeight, shadowLayers,
                            shadowMipLevels);
                    } else {
                        shadowMaps.GetDimensions(
                            0, shadowWidth, shadowHeight, shadowLayers,
                            shadowMipLevels);
                    }
                    float2 texelSize =
                        1.0f / float2(shadowWidth, shadowHeight);
                    float2 minSampleCoordinate = texelSize * 0.5f;
                    float2 maxSampleCoordinate = 1.0f - minSampleCoordinate;
                    for (int sampleIndex = 0; sampleIndex < 4;
                         ++sampleIndex) {
                        float2 sampleCoordinate = clamp(
                            shadowCoordinates.xy +
                                shadowPoissonDisk[sampleIndex] * texelSize,
                            minSampleCoordinate, maxSampleCoordinate);
                        float storedDepth =
                            light.lightType == 2
                                ? globalShadowMaps.SampleLevel(
                                      materialSampler,
                                      float3(sampleCoordinate,
                                             float(light.lightShadowMapIndex)),
                                      0.0f)
                                      .r
                                : shadowMaps.SampleLevel(
                                      materialSampler,
                                      float3(sampleCoordinate,
                                             float(light.lightShadowMapIndex)),
                                      0.0f)
                                      .r;
                        visibleSamples +=
                            shadowCoordinates.z - bias <= storedDepth
                                ? 1.0f
                                : 0.0f;
                    }
                    visibility =
                        lerp(0.12f, 1.0f, visibleSamples * 0.25f);
                }
            }
        }

        float3 H = normalize(V + L);
        float3 F = fresnelSchlick(max(dot(H, V), 0.0f), F0);
        float NDF = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, L, roughness);
        float3 specular =
            (NDF * G * F) /
            max(4.0f * max(dot(N, V), 0.0f) * nDotL, 0.001f);
        float3 kS = F;
        float3 kD = (1.0f.xxx - kS) * (1.0f - metallic);
        float3 radiance = light.lightIntensity * attenuation * light.lightColor.rgb;
        directLighting += (kD * albedo / 3.14159265f + specular) *
                          radiance * nDotL * visibility;
    }

    float3 ambient = 0.03f * albedo;
    if (materialParameters.w > 0.0f) {
        float3 irradiance = environmentMap.SampleLevel(
            materialSampler, float4(N, 0.0f), 1.0f).rgb;
        float3 reflectionDirection = reflect(-V, N);
        float3 prefilteredRadiance = environmentMap.SampleLevel(
            materialSampler, float4(reflectionDirection, 1.0f),
            roughness * 9.0f).rgb;
        float3 ambientFresnel =
            F0 +
            (max((1.0f - roughness).xxx, F0) - F0) *
                pow(1.0f - max(dot(N, V), 0.0f), 5.0f);
        uint brdfWidth;
        uint brdfHeight;
        uint brdfMipLevels;
        brdfLUT.GetDimensions(0, brdfWidth, brdfHeight, brdfMipLevels);
        float2 brdfTexelSize = 1.0f / float2(brdfWidth, brdfHeight);
        float2 brdfCoordinates = clamp(
            float2(max(dot(N, V), 0.0f), roughness),
            brdfTexelSize * 0.5f, 1.0f - brdfTexelSize * 0.5f);
        float2 environmentBRDF =
            brdfLUT.Sample(materialSampler, brdfCoordinates).rg;
        float3 diffuseWeight =
            (1.0f.xxx - ambientFresnel) * (1.0f - metallic);
        float3 diffuseEnvironment = irradiance * albedo * diffuseWeight;
        float3 specularEnvironment =
            prefilteredRadiance *
            (ambientFresnel * environmentBRDF.x + environmentBRDF.y);
        ambient = (diffuseEnvironment + specularEnvironment) *
                  ambientOcclusion;
    } else {
        ambient *= ambientOcclusion;
    }
    float3 color = ambient + directLighting;
    color = color / (color + 1.0f);
    if (output_is_srgb == 0) {
        color = pow(max(color, 0.0f.xxx), 1.0f / 2.2f);
    }
    return float4(color, baseColor.a * albedoSample.a);
}
