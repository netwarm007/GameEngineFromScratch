#include "functions.h.hlsl"

RWTexture2D<float2> outputTexture : register(u0);

MYGE_ROOT_SIGNATURE("DescriptorTable(UAV(u0, numDescriptors=1))")
[numthreads(2, 2, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float2 ndc = float2(id.x == 0 ? -0.6f : 0.6f,
                        id.y == 0 ? 0.5f : -0.5f);
    outputTexture[id.xy] = shadow_map_coordinates(float4(ndc, 0.4f, 1.0f), 1).xy;
}
