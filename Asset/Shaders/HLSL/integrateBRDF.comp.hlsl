#include "functions.h.hlsl"

#define BrdfRS  "RootFlags(0)," \
                "DescriptorTable( UAV(u0, numDescriptors = 1, " \
                               "        flags = DESCRIPTORS_VOLATILE))"

RWTexture2D<float4> img_output : register(u0);

float2 IntegrateBRDF(float NdotV, float roughness)
{
    float3 V;
    V.x = sqrt(1.0 - NdotV*NdotV);
    V.y = 0.0;
    V.z = NdotV;

    float A = 0.0;
    float B = 0.0;

    float3 N = float3(0.0, 0.0, 1.0);

    const uint SAMPLE_COUNT = 1024u;
    for(uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        float2 Xi = Hammersley(i, SAMPLE_COUNT);
        float3 H  = ImportanceSampleGGX(Xi, N, roughness);
        float3 L  = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        if(NdotL > 0.0)
        {
            float G = GeometrySmithIndirect(N, V, L, roughness);
            float G_Vis = (G * VdotH) / (NdotH * NdotV);
            float Fc = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    A /= float(SAMPLE_COUNT);
    B /= float(SAMPLE_COUNT);
    return float2(A, B);
}

[numthreads(1, 1, 1)]
[RootSignature(BrdfRS)]
void integrateBRDF_comp_main(uint3 DTid : SV_DISPATCHTHREADID)
{
    float4 pixel;
    int2 pixel_coords = int2(DTid.xy);
    pixel.rg = IntegrateBRDF(float(pixel_coords.x + 1) / 512.0f, float(pixel_coords.y + 1) / 512.0f);
    img_output[pixel_coords] = pixel.rgrg; // repeat rg channel to fix HLSL -> Metal conversion bug
}
