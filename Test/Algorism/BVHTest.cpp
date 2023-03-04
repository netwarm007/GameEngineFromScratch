#include <cstdio>
#include <iostream>

#include "BVH.hpp"

#include "TestScene.hpp"

int main() {
    auto scene = random_scene();

    My::BVHNode<float_precision> root(scene);

    std::cout << root << std::endl;

    return 0;
}