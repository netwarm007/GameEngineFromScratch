#include "integrateBRDF.comp.hlsl"

[numthreads(8, 8, 1)]
MYGE_ROOT_SIGNATURE(BrdfRS)
void vulkan_integrateBRDF_comp_main(uint3 DTid : SV_DISPATCHTHREADID)
{
    if (DTid.x >= 512 || DTid.y >= 512) return;
    float2 integrated =
        IntegrateBRDF(float(DTid.x + 1) / 512.0f,
                      float(DTid.y + 1) / 512.0f);
    img_output[DTid.xy] = integrated.rgrg;
}
