#include "cbuffer.h"

#define MipmapRS "RootFlags(0), " \
    "DescriptorTable(SRV(t0, numDescriptors=1), UAV(u0, numDescriptors=1))"

Texture2D<float4> sourceTexture : register(t0);
RWTexture2D<float4> destinationTexture : register(u0);
[numthreads(8, 8, 1)]
MYGE_ROOT_SIGNATURE(MipmapRS)
void generate_mips_comp_main(uint3 id : SV_DispatchThreadID) {
    uint width, height;
    destinationTexture.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) return;
    uint sourceWidth, sourceHeight;
    sourceTexture.GetDimensions(sourceWidth, sourceHeight);
    float2 scale = float2(sourceWidth, sourceHeight) / float2(width, height);
    float2 low = float2(id.xy) * scale;
    float2 high = float2(id.xy + 1) * scale;
    uint2 begin = uint2(floor(low));
    uint2 end = min(uint2(ceil(high)), uint2(sourceWidth, sourceHeight));
    float4 sum = 0.0f;
    for (uint y = begin.y; y < end.y; ++y) {
        float weightY = min(high.y, float(y + 1)) - max(low.y, float(y));
        for (uint x = begin.x; x < end.x; ++x) {
            float weightX = min(high.x, float(x + 1)) - max(low.x, float(x));
            sum += sourceTexture.Load(int3(x, y, 0)) * weightX * weightY;
        }
    }
    destinationTexture[id.xy] = sum / (scale.x * scale.y);
}
