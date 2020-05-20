#include "cbuffer.h"
[RootSignature(MyRS1)]
float4 shadowmap_frag_main() : SV_Target
{
    // this shader should be never executed
    return 1.0f;
}