#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct shadowmap_frag_main_out
{
    float4 _entryPointOutput [[color(0)]];
};

float4 _shadowmap_frag_main()
{
    return float4(1.0);
}

fragment shadowmap_frag_main_out shadowmap_frag_main()
{
    shadowmap_frag_main_out out = {};
    out._entryPointOutput = _shadowmap_frag_main();
    return out;
}

