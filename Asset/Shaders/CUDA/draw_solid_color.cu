#include <optix.h>

#include "OptixTest.h"
#include "ColorSpaceConversion.hpp"

using vec3 = My::Vector3f;
using color = My::RGBf;
using ray  = My::Ray<float>;

extern "C" {
__constant__ Params params;
}

__inline__ __device__ vec3 _V(float3 f) {
    return vec3({f.x, f.y, f.z});
}

extern "C"
__global__ void __raygen__draw_solid_color() {
    uint3 launch_index = optixGetLaunchIndex();
    RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();

    unsigned int i = launch_index.x;
    unsigned int j = launch_index.y;
    unsigned int pixel_index = j * params.image->Width + i;

    curandState* local_rand_state = &params.rand_state[pixel_index];

    float u = float(i + curand_uniform(local_rand_state)) / params.image->Width;
    float v = float(j + curand_uniform(local_rand_state)) / params.image->Height;
    ray r = params.cam->get_ray(u, v, local_rand_state);

    vec3 unit_direction = r.getDirection();
    float t = 0.5f * (unit_direction[1] + 1.0f);
    vec3 c = (1.0f - t) * color({0, 0, 0}) + t * color({rtData->r, rtData->g, rtData->b});

    vec3* pOutputBuffer = reinterpret_cast<vec3*>(params.image->data);
    pOutputBuffer[pixel_index] = My::Linear2SRGB(c);
}
