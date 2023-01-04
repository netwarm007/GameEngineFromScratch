#include <curand_kernel.h>
#include "geommath.hpp"
#include "RayTracingCamera.hpp"
#include "Image.hpp"
#include "Color.hpp"

struct Params{
    My::Image*                      image;
    My::RayTracingCamera<float>*    cam;
    curandStateMRG32k3a*                    rand_state;
    OptixTraversableHandle          handle;
};

struct RayGenData{
    int num_of_samples;
};

struct MissData {
    My::RGBf bg_color;
};

struct HitGroupData {
    // No data needed
};