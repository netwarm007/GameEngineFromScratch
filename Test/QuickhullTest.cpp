#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include "quickhull.hpp"

using namespace std;
using namespace My;

using TestDataType = float;

int main(int argc, char** argv) {
    int point_num = 30;

    if (argc > 1) {
        point_num = atoi(argv[1]);
    }

    default_random_engine generator;
    uniform_real_distribution<TestDataType> distribution(-1.0, 1.0);
    auto dice = std::bind(distribution, generator);

    QuickHull<TestDataType> quick_hull;
    PointSet<TestDataType> point_set;
    cout << "Points Generated:" << endl;
    for (auto i = 0; i < point_num; i++) {
        PointPtr<TestDataType> point_ptr = make_shared<Point<TestDataType>>(Point<TestDataType>{dice(), dice(), dice()});
        cout << *point_ptr;
        point_set.insert(std::move(point_ptr));
    }

    Polyhedron<TestDataType> convex_hull;
    while (quick_hull.Iterate(convex_hull, point_set)) {
        cerr << "num of faces after this iteration: "
             << convex_hull.Faces.size() << endl;
    }

    cerr << "num of faces generated: " << convex_hull.Faces.size() << endl;

    return 0;
}
