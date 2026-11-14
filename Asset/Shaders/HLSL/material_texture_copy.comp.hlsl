Texture2D<float4> sourceTexture : register(t0);
RWTexture2D<float4> destinationTexture : register(u1);
SamplerState sourceSampler : register(s0);

[numthreads(8, 8, 1)]
void material_texture_copy_comp_main(uint3 id : SV_DISPATCHTHREADID) {
    uint width, height;
    destinationTexture.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) return;
    float2 uv = (float2(id.xy) + 0.5f) / float2(width, height);
    destinationTexture[id.xy] = sourceTexture.SampleLevel(sourceSampler, uv, 0.0f);
}
