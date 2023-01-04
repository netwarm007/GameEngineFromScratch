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

struct MyTracePayload {
    vec3 attenuation;
    vec3 scatter_ray_origin;
    vec3 scatter_ray_direction;
    int  max_depth;
    bool done;
};

static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        vec3                   ray_origin,
        vec3                   ray_direction,
        float                  tmin,
        float                  tmax,
        MyTracePayload&        prd ) {
    unsigned int p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10;
    p0 = __float_as_uint(prd.attenuation[0]);
    p1 = __float_as_uint(prd.attenuation[1]);
    p2 = __float_as_uint(prd.attenuation[2]);
    p3 = __float_as_uint(prd.scatter_ray_origin[0]);
    p4 = __float_as_uint(prd.scatter_ray_origin[1]);
    p5 = __float_as_uint(prd.scatter_ray_origin[2]);
    p6 = __float_as_uint(prd.scatter_ray_direction[0]);
    p7 = __float_as_uint(prd.scatter_ray_direction[1]);
    p8 = __float_as_uint(prd.scatter_ray_direction[2]);
    p9 = prd.max_depth;
    p10 = (prd.done) ? 1 : 0;

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
            p0, p1, p2,
            p3, p4, p5,
            p6, p7, p8, 
            p9, p10 );
    prd.attenuation[0] = __uint_as_float(p0);
    prd.attenuation[1] = __uint_as_float(p1);
    prd.attenuation[2] = __uint_as_float(p2);
    prd.scatter_ray_origin[0] = __uint_as_float(p3);
    prd.scatter_ray_origin[1] = __uint_as_float(p4);
    prd.scatter_ray_origin[2] = __uint_as_float(p5);
    prd.scatter_ray_direction[0] = __uint_as_float(p6);
    prd.scatter_ray_direction[1] = __uint_as_float(p7);
    prd.scatter_ray_direction[2] = __uint_as_float(p8);
    prd.max_depth = p9;
    prd.done = (p10 == 1) ? true : false;
}

static __forceinline__ __device__ void setPayload( const vec3& attenuation, bool is_scattered)
{
    optixSetPayload_0( __float_as_uint( attenuation[0] ) );
    optixSetPayload_1( __float_as_uint( attenuation[1] ) );
    optixSetPayload_2( __float_as_uint( attenuation[2] ) );
    optixSetPayload_10( is_scattered ? 0 : 1 );
}

static __forceinline__ __device__ void setPayload( const vec3& attenuation, const ray& ray_scattered, bool is_scattered, int max_depth )
{
    auto orig   = ray_scattered.getOrigin();
    auto direct = ray_scattered.getDirection();
    optixSetPayload_0( __float_as_uint( attenuation[0] ) );
    optixSetPayload_1( __float_as_uint( attenuation[1] ) );
    optixSetPayload_2( __float_as_uint( attenuation[2] ) );
    optixSetPayload_3( __float_as_uint( orig[0] ) );
    optixSetPayload_4( __float_as_uint( orig[1] ) );
    optixSetPayload_5( __float_as_uint( orig[2] ) );
    optixSetPayload_6( __float_as_uint( direct[0] ) );
    optixSetPayload_7( __float_as_uint( direct[1] ) );
    optixSetPayload_8( __float_as_uint( direct[2] ) );
    optixSetPayload_9( max_depth );
    optixSetPayload_10( is_scattered ? 0 : 1 );
}

static __forceinline__ __device__ MyTracePayload getPayload()
{
    return MyTracePayload({
            {
                __uint_as_float( optixGetPayload_0() ),
                __uint_as_float( optixGetPayload_1() ),
                __uint_as_float( optixGetPayload_2() )
            },
            {
                __uint_as_float( optixGetPayload_3() ),
                __uint_as_float( optixGetPayload_4() ),
                __uint_as_float( optixGetPayload_5() )
            },
            {
                __uint_as_float( optixGetPayload_6() ),
                __uint_as_float( optixGetPayload_7() ),
                __uint_as_float( optixGetPayload_8() )
            },
                static_cast<int>(optixGetPayload_9()),
                optixGetPayload_10() ? true : false
            });
}

extern "C"
__global__ void __raygen__rg() {
    uint3 launch_index = optixGetLaunchIndex();
    // RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer(); 

    unsigned int i = launch_index.x;
    unsigned int j = launch_index.y;
    unsigned int pixel_index = j * params.image->Width + i;

    curandStateMRG32k3a* local_rand_state = &params.rand_state[pixel_index];

    int num_of_samples = params.num_of_samples;
    vec3 col = {0.f, 0.f, 0.f};

    for (int s = 0; s < num_of_samples; s++) {
        float u = float(i + curand_uniform(local_rand_state)) / params.image->Width;
        float v = float(j + curand_uniform(local_rand_state)) / params.image->Height;
        ray r = params.cam->get_ray(u, v, local_rand_state);

        vec3 attenuation = {1.f, 1.f, 1.f};
        MyTracePayload payload;
        payload.attenuation = attenuation;
        payload.scatter_ray_origin = r.getOrigin(); 
        payload.scatter_ray_direction = r.getDirection(); 
        payload.max_depth = params.max_depth; 
        payload.done = false; 

        do {
            trace( params.handle,
                    payload.scatter_ray_origin,
                    payload.scatter_ray_direction,
                    0.001f,   // tmin
                    FLT_MAX,  // tmax
                    payload);

            attenuation = attenuation * payload.attenuation;
        } while (!payload.done);

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

    setPayload(c, false);
}

extern "C"
__global__ void __closesthit__ch() {
    auto payload = getPayload();
    if(payload.max_depth < 0) {
        setPayload({0.f, 0.f, 0.f}, false);
        return;
    }

    uint3 launch_index = optixGetLaunchIndex();

    float t_hit = optixGetRayTmax();
    unsigned int i = launch_index.x;
    unsigned int j = launch_index.y;
    unsigned int pixel_index = j * params.image->Width + i;

    curandStateMRG32k3a* local_rand_state = &params.rand_state[pixel_index];

    const vec3 ray_orig   = _V(optixGetWorldRayOrigin());
    const vec3 ray_dir    = _V(optixGetWorldRayDirection());

    const unsigned int              prim_idx    = optixGetPrimitiveIndex();
    const OptixTraversableHandle    gas         = optixGetGASTraversableHandle();
    const unsigned int              sbtGASIndex = optixGetSbtGASIndex();

    float4 q;
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    optixGetSphereData(gas, prim_idx, 0, 0.f, &q);

    vec3 world_raypos = ray_orig + t_hit * ray_dir;
    vec3 obj_raypos   = _V(optixTransformPointFromWorldToObjectSpace(_f(world_raypos)));
    vec3 obj_normal   = (obj_raypos - _V({q.x, q.y, q.z})) / q.w;
    vec3 world_normal = _V(optixTransformNormalFromObjectToWorldSpace(_f(obj_normal)));
    My::Normalize(world_normal);

    ray scattered;
    ray r_in(ray_orig, ray_dir);
    hit_record rec;
    rec.set(t_hit, world_raypos, world_normal, optixIsFrontFaceHit(), nullptr);
    color attenuation;
    HitGroupData* hg_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer() + sbtGASIndex * sizeof(HitGroupSbtRecord));
    bool b_scattered;
    switch(hg_data->material_type) {
        case Material::MAT_DIFFUSE:
        {
            b_scattered = lambertian::scatter_static(r_in, rec, attenuation, scattered, local_rand_state, hg_data->base_color);
        }
        break;
        case Material::MAT_METAL:
        {
            b_scattered = metal::scatter_static(r_in, rec, attenuation, scattered, local_rand_state, hg_data->base_color, hg_data->fuzz);
        }
        break;
        case Material::MAT_DIELECTRIC:
        {
            b_scattered = dielectric::scatter_static(r_in, rec, attenuation, scattered, local_rand_state, hg_data->ir);
        }
        break;
    }

    setPayload(attenuation, scattered, b_scattered, payload.max_depth - 1);
}