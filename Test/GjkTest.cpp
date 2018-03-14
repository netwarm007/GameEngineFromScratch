#include <functional>
#include <iostream>
#include <random>
#include "quickhull.hpp"
#include "Gjk.hpp"

using namespace My;
using namespace std;
using namespace std::placeholders;

int main(int argc, char** argv)
{
    int point_num = 100;

    if(argc > 1)
    {
        point_num = atoi(argv[1]);
    }

    Polyhedron A, B;
    {
        cout << "Generate Polyhedron A" << endl;
        default_random_engine generator;
        generator.seed(1);
        uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        auto dice = std::bind(distribution, generator);

        QuickHull quick_hull;
        cout << "Points Generated:" << endl;
        for(auto i = 0; i < point_num; i++)
        {
            PointPtr point_ptr = make_shared<Point>(dice(), dice(), dice());
            quick_hull.AddPoint(std::move(point_ptr));
        }

        quick_hull.Init();
        while (quick_hull.Iterate())
            ;

        A = quick_hull.GetHull();
        cerr << "num of faces generated: " << A.Faces.size() << endl;
    }

    {
        cout << "Generate Polyhedron B" << endl;
        default_random_engine generator;
        generator.seed(200);
        uniform_real_distribution<float> distribution(0.6f, 1.7f);
        auto dice = std::bind(distribution, generator);

        QuickHull quick_hull;
        cout << "Points Generated:" << endl;
        for(auto i = 0; i < point_num; i++)
        {
            PointPtr point_ptr = make_shared<Point>(dice(), dice(), dice());
            quick_hull.AddPoint(std::move(point_ptr));
        }

        quick_hull.Init();
        while (quick_hull.Iterate())
            ;

        B = quick_hull.GetHull();
        cerr << "num of faces generated: " << B.Faces.size() << endl;
    }

    SupportFunction support_function_A = std::bind(ConvexPolyhedronSupportFunction, A, _1);
    SupportFunction support_function_B = std::bind(ConvexPolyhedronSupportFunction, B, _1);
    PointList simplex;
    Vector3f direction (1.0f, 0.0f, 0.0f);
    int intersected;
    while ((intersected = GjkIntersection(support_function_A, support_function_B, direction, simplex)) == -1)
        cerr << "approximate direction: " << direction;

    switch (intersected)
    {
    case 1:
        cout << "A and B IS intersected" << endl;
        break;
    case 0:
        cout << "A and B is NOT intersected" << endl;
        break;
    default:
        assert(0);
    }

    return 0;
}