#include <optix.h>

#include "OptixTest.hpp"
#include "ColorSpaceConversion.hpp"

using vec3 = My::Vector3f;
using color = My::RGBf;
using ray  = My::Ray<float>;

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ vec3 _V(float3 f) {
    return vec3({f.x, f.y, f.z});
}

static __forceinline__ __device__ float3 _f(vec3 v) {
    return make_float3(v[0], v[1], v[2]);
}

static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        vec3                   ray_origin,
        vec3                   ray_direction,
        float                  tmin,
        float                  tmax,
        vec3&                  prd ) {
    unsigned int p0, p1, p2;
    p0 = __float_as_uint(prd[0]);
    p1 = __float_as_uint(prd[1]);
    p0 = __float_as_uint(prd[2]);
    optixTrace(
            handle,
            _f(ray_origin),
            _f(ray_direction),
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset
            0,                   // SBT stride
            0,                   // missSBTIndex
            p0, p1, p2 );
    prd[0] = __uint_as_float(p0);
    prd[1] = __uint_as_float(p1);
    prd[2] = __uint_as_float(p2);
}


static __forceinline__ __device__ void setPayload( vec3 p )
{
    optixSetPayload_0( __float_as_uint( p[0] ) );
    optixSetPayload_1( __float_as_uint( p[1] ) );
    optixSetPayload_2( __float_as_uint( p[2] ) );
}


static __forceinline__ __device__ vec3 getPayload()
{
    return vec3({
                __uint_as_float( optixGetPayload_0() ),
                __uint_as_float( optixGetPayload_1() ),
                __uint_as_float( optixGetPayload_2() )
            });
}

extern "C"
__global__ void __raygen__rg() {
    uint3 launch_index = optixGetLaunchIndex();
    RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();

    unsigned int i = launch_index.x;
    unsigned int j = launch_index.y;
    unsigned int pixel_index = j * params.image->Width + i;

    curandState* local_rand_state = &params.rand_state[pixel_index];

    int num_of_samples = rtData->num_of_samples;
    vec3 col = {0.f, 0.f, 0.f};

    for (int s = 0; s < num_of_samples; s++) {
        float u = float(i + curand_uniform(local_rand_state)) / params.image->Width;
        float v = float(j + curand_uniform(local_rand_state)) / params.image->Height;
        ray r = params.cam->get_ray(u, v, local_rand_state);

        vec3 attenuation = {1.0f, 1.0f, 1.0f};
        trace( params.handle,
                r.getOrigin(),
                r.getDirection(),
                0.00f,  // tmin
                FLT_MAX,  // tmax
                attenuation);

        col += attenuation;
    }

    col = col / (float)num_of_samples;

    vec3* pOutputBuffer = reinterpret_cast<vec3*>(params.image->data);
    pOutputBuffer[pixel_index] = My::Linear2SRGB(col);
}

extern "C"
__global__ void __miss__ms() {
    MissData* missData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());

    vec3 unit_direction = _V(optixGetWorldRayDirection());
    float t = 0.5f * (unit_direction[1] + 1.0f);
    vec3 c = (1.0f - t) * color({0, 0, 0}) + t * (missData->bg_color);

    setPayload(c);
}

extern "C"
__global__ void __closesthit__ch() {
    float t_hit = optixGetRayTmax();

    const vec3 ray_orig   = _V(optixGetWorldRayOrigin());
    const vec3 ray_dir    = _V(optixGetWorldRayDirection());

    const unsigned int              prim_idx    = optixGetPrimitiveIndex();
    const OptixTraversableHandle    gas         = optixGetGASTraversableHandle();
    const unsigned int              sbtGASIndex = optixGetSbtGASIndex();

    float4 q;
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    optixGetSphereData(gas, prim_idx, sbtGASIndex, 0.f, &q);

    vec3 world_raypos = ray_orig + t_hit * ray_dir;
    vec3 obj_raypos   = _V(optixTransformPointFromWorldToObjectSpace(_f(world_raypos)));
    vec3 obj_normal   = (obj_raypos - _V({q.x, q.y, q.z})) / q.w;
    vec3 world_normal = _V(optixTransformNormalFromObjectToWorldSpace(_f(obj_normal)));
    My::Normalize(world_normal);

    setPayload(world_normal * 0.5f + 0.5f);
}