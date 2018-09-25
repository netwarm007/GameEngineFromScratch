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

layout(location = 0) in highp vec2 UV;
layout(location = 0) out highp vec2 FragColor;

highp float RadicalInverse_VdC(inout uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 1431655765u) << 1u) | ((bits & 2863311530u) >> 1u);
    bits = ((bits & 858993459u) << 2u) | ((bits & 3435973836u) >> 2u);
    bits = ((bits & 252645135u) << 4u) | ((bits & 4042322160u) >> 4u);
    bits = ((bits & 16711935u) << 8u) | ((bits & 4278255360u) >> 8u);
    return float(bits) * 2.3283064365386962890625e-10;
}

highp vec2 Hammersley(uint i, uint N)
{
    uint param = i;
    highp float _156 = RadicalInverse_VdC(param);
    return vec2(float(i) / float(N), _156);
}

highp vec3 ImportanceSampleGGX(highp vec2 Xi, highp vec3 N, highp float roughness)
{
    highp float a = roughness * roughness;
    highp float phi = 6.283185482025146484375 * Xi.x;
    highp float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (((a * a) - 1.0) * Xi.y)));
    highp float sinTheta = sqrt(1.0 - (cosTheta * cosTheta));
    highp vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;
    highp vec3 up = mix(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), bvec3(abs(N.z) < 0.999000012874603271484375));
    highp vec3 tangent = normalize(cross(up, N));
    highp vec3 bitangent = cross(N, tangent);
    highp vec3 sampleVec = ((tangent * H.x) + (bitangent * H.y)) + (N * H.z);
    return normalize(sampleVec);
}

highp float GeometrySchlickGGXIndirect(highp float NdotV, highp float roughness)
{
    highp float a = roughness;
    highp float k = (a * a) / 2.0;
    highp float nom = NdotV;
    highp float denom = (NdotV * (1.0 - k)) + k;
    return nom / denom;
}

highp float GeometrySmithIndirect(highp vec3 N, highp vec3 V, highp vec3 L, highp float roughness)
{
    highp float NdotV = max(dot(N, V), 0.0);
    highp float NdotL = max(dot(N, L), 0.0);
    highp float param = NdotV;
    highp float param_1 = roughness;
    highp float ggx2 = GeometrySchlickGGXIndirect(param, param_1);
    highp float param_2 = NdotL;
    highp float param_3 = roughness;
    highp float ggx1 = GeometrySchlickGGXIndirect(param_2, param_3);
    return ggx1 * ggx2;
}

highp vec2 IntegrateBRDF(highp float NdotV, highp float roughness)
{
    highp vec3 V;
    V.x = sqrt(1.0 - (NdotV * NdotV));
    V.y = 0.0;
    V.z = NdotV;
    highp float A = 0.0;
    highp float B = 0.0;
    highp vec3 N = vec3(0.0, 0.0, 1.0);
    for (uint i = 0u; i < 1024u; i++)
    {
        uint param = i;
        uint param_1 = 1024u;
        highp vec2 Xi = Hammersley(param, param_1);
        highp vec2 param_2 = Xi;
        highp vec3 param_3 = N;
        highp float param_4 = roughness;
        highp vec3 H = ImportanceSampleGGX(param_2, param_3, param_4);
        highp vec3 L = normalize((H * (2.0 * dot(V, H))) - V);
        highp float NdotL = max(L.z, 0.0);
        highp float NdotH = max(H.z, 0.0);
        highp float VdotH = max(dot(V, H), 0.0);
        if (NdotL > 0.0)
        {
            highp vec3 param_5 = N;
            highp vec3 param_6 = V;
            highp vec3 param_7 = L;
            highp float param_8 = roughness;
            highp float G = GeometrySmithIndirect(param_5, param_6, param_7, param_8);
            highp float G_Vis = (G * VdotH) / (NdotH * NdotV);
            highp float Fc = pow(1.0 - VdotH, 5.0);
            A += ((1.0 - Fc) * G_Vis);
            B += (Fc * G_Vis);
        }
    }
    A /= 1024.0;
    B /= 1024.0;
    return vec2(A, B);
}

void main()
{
    highp float param = UV.x;
    highp float param_1 = UV.y;
    highp vec2 integratedBRDF = IntegrateBRDF(param, param_1);
    FragColor = integratedBRDF;
}

