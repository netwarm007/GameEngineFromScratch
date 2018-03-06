#include <iostream>
#include <random>
#include <limits>
#include "quickhull.hpp"

using namespace std;
using namespace My;

int main(int argc, char** argv)
{
    int point_num = 10;

    if(argc > 1)
    {
        point_num = atoi(argv[1]);
    }

    default_random_engine generator;
    uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    auto dice = std::bind(distribution, generator);

    QuickHull quick_hull;
    cout << "Points Generated:" << endl;
    for(auto i = 0; i < point_num; i++)
    {
        PointPtr point_ptr = make_shared<Point>(dice(), dice(), dice());
        cout << *point_ptr;
        quick_hull.AddPoint(std::move(point_ptr));
    }

    quick_hull.ComputeHull();
    auto convex_hull = quick_hull.GetHull();

    cout << "Points On the Convex Hull:" << endl;
    for (auto pFace : convex_hull.Faces)
    {
        cout << "Face:" << endl;
        for (auto pEdge : pFace->Edges)
            cout << *pEdge->first << "--" << *pEdge->second;
    }

    return 0;
}