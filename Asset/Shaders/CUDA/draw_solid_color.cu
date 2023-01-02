#include <optix.h>

#include "OptixTest.h"

/** clamp */
__forceinline__ __device__ float clamp( const float f, const float a, const float b )
{
    return fmaxf( a, fminf( f, b ) );
}

__forceinline__ __device__ float3 clamp(const float3& v, const float a, const float b)
{
  return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

__forceinline__ __device__ float3 toSRGB( const float3& c ) {
    float  invGamma = 1.0f / 2.4f;
    float3 powed    = make_float3( powf( c.x, invGamma ), powf( c.y, invGamma ), powf( c.z, invGamma ) );
    return make_float3(
        c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
        c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
        c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f );
}

__forceinline__ __device__ unsigned char quantizeUnsigned8Bits( float x) {
   x = clamp( x, 0.0f, 1.0f );
   enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
   return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}

__forceinline__ __device__ uchar4 make_color( const float3& c ) {
    // first apply gamma, then convert to unsigned char
    float3 srgb = toSRGB( clamp( c, 0.0f, 1.0f ) );
    return make_uchar4( quantizeUnsigned8Bits( srgb.x ), quantizeUnsigned8Bits( srgb.y ), quantizeUnsigned8Bits( srgb.z ), 255u );
}

extern "C" {
__constant__ Params params;
}

extern "C"
__global__ void __raygen__draw_solid_color() {
    uint3 launch_index = optixGetLaunchIndex();
    RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();
    params.image[launch_index.y * params.image_width + launch_index.x] =
        make_color( make_float3( rtData->r, rtData->g, rtData->b ) );
}
