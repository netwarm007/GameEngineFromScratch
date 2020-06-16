#include "functions.h.hlsl"

RWTexture2D<float4> img_output : register(u0);

float4 RayTrace(int2 pixel_coords)
{
    return float4(float(pixel_coords.x) / 512.0f, float(pixel_coords.y) / 512.0f, 0.3f, 1.0f);
}

[numthreads(1, 1, 1)]
void raytrace_comp_main(uint3 DTid : SV_DISPATCHTHREADID)
{
    float4 pixel;
    int2 pixel_coords = int2(DTid.xy);
    pixel.rgba = RayTrace(pixel_coords);
    img_output[pixel_coords] = pixel.rgba;
}