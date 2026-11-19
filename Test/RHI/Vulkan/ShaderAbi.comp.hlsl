#include "cbuffer.h"

RWTexture2D<float4> outputTexture : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float4 position = mul(float4(1, 2, 3, 1), modelMatrix);
    position = mul(mul(position, viewMatrix), projectionMatrix);
    outputTexture[uint2(0, 0)] = position;
    outputTexture[uint2(1, 0)] =
        float4(camPos.xyz, numLights + clip_space_type);
    outputTexture[uint2(0, 1)] =
        float4(lights[1].lightPosition.xyz, lights[1].lightIntensity);
    outputTexture[uint2(1, 1)] =
        mul(float4(light_index, shadowmap_layer_index, near_plane, far_plane),
            shadowMatrices[5]);
}
