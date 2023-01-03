#include <curand_kernel.h>
#include "geommath.hpp"
#include "RayTracingCamera.hpp"
#include "Image.hpp"

struct Params
{
    My::Image*                      image;
    My::RayTracingCamera<float>*    cam;
    curandState*                    rand_state;
};

struct RayGenData
{
    float r,g,b;
};
