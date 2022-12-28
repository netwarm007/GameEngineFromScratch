#include <memory>

#include "TestMaterial.hpp"
#include "Sphere.hpp"
#include "HitableList.hpp"

// World
auto random_scene() {
    My::HitableList<float_precision> world;

    auto material_ground = std::make_shared<lambertian>(color({0.5, 0.5, 0.5}));
    world.add(std::make_shared<My::Sphere<float_precision, std::shared_ptr<material>>>(
        (float_precision)1000.0, point3({0, -1000, -1.0}), material_ground));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = My::random_f<float_precision>();
            point3 center(
                {a + (float_precision)0.9 * My::random_f<float_precision>(),
                 (float_precision)0.2,
                 b + (float_precision)0.9 * My::random_f<float_precision>()});

            if (Length(center - point3({4, 0.2, 0})) > 0.9) {
                std::shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    color albedo = My::random_v<float_precision, 3>() *
                                   My::random_v<float_precision, 3>();
                    sphere_material = std::make_shared<lambertian>(albedo);
                    world.add(
                        std::make_shared<My::Sphere<float_precision, std::shared_ptr<material>>>(
                            (float_precision)0.2, center, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = My::random_v<float_precision, 3>(0.5, 1);
                    auto fuzz = My::random_f<float_precision>(0, 0.5);
                    sphere_material = std::make_shared<metal>(albedo, fuzz);
                    world.add(
                        std::make_shared<My::Sphere<float_precision, std::shared_ptr<material>>>(
                            (float_precision)0.2, center, sphere_material));
                } else {
                    // glass
                    sphere_material = std::make_shared<dielectric>((float_precision)1.5);
                    world.add(
                        std::make_shared<My::Sphere<float_precision, std::shared_ptr<material>>>(
                            (float_precision)0.2, center, sphere_material));
                }
            }
        }
    }

    auto material_1 = std::make_shared<dielectric>((float_precision)1.5);
    world.add(std::make_shared<My::Sphere<float_precision, std::shared_ptr<material>>>(
        (float_precision)1.0, point3({0, 1, 0}), material_1));

    auto material_2 = std::make_shared<lambertian>(color({0.4, 0.2, 0.1}));
    world.add(std::make_shared<My::Sphere<float_precision, std::shared_ptr<material>>>(
        (float_precision)1.0, point3({-4, 1, 0}), material_2));

    auto material_3 = std::make_shared<metal>(color({0.7, 0.6, 0.5}), (float_precision)0.1);
    world.add(std::make_shared<My::Sphere<float_precision, std::shared_ptr<material>>>(
        (float_precision)1.0, point3({4, 1, 0}), material_3));

    return world;
}
