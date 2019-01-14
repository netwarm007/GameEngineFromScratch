#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct u_lightParams
{
    float3 u_lightParams_lightPos;
    float u_lightParams_far_plane;
};

fragment void shadowmap_omni_frag_main()
{
    float _gl_FragDepth;
    float4 FragPos;
}

