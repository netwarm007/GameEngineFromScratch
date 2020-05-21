#include "quickhull.hpp"

using namespace My;
using namespace std;

bool QuickHull::Init(Polyhedron& hull, PointSet& point_set) {
    if (point_set.size() < 4) {
        // too few points in the point set, nothing could be done
        return false;
    }

    PointPtr ExtremePointXMin =
        make_shared<Point>(numeric_limits<float>::max());
    PointPtr ExtremePointYMin =
        make_shared<Point>(numeric_limits<float>::max());
    PointPtr ExtremePointZMin =
        make_shared<Point>(numeric_limits<float>::max());
    PointPtr ExtremePointXMax =
        make_shared<Point>(numeric_limits<float>::lowest());
    PointPtr ExtremePointYMax =
        make_shared<Point>(numeric_limits<float>::lowest());
    PointPtr ExtremePointZMax =
        make_shared<Point>(numeric_limits<float>::lowest());

    // finding the Extreme Points [O(n) complexity]
    for (const auto& point_ptr : point_set) {
        if (point_ptr->data[0] < ExtremePointXMin->data[0])
            ExtremePointXMin = point_ptr;
        if (point_ptr->data[1] < ExtremePointYMin->data[1])
            ExtremePointYMin = point_ptr;
        if (point_ptr->data[2] < ExtremePointZMin->data[2])
            ExtremePointZMin = point_ptr;
        if (point_ptr->data[0] > ExtremePointXMax->data[0])
            ExtremePointXMax = point_ptr;
        if (point_ptr->data[1] > ExtremePointYMax->data[1])
            ExtremePointYMax = point_ptr;
        if (point_ptr->data[2] > ExtremePointZMax->data[2])
            ExtremePointZMax = point_ptr;
    }

    PointSet ExtremePoints;
    ExtremePoints.insert({ExtremePointXMin, ExtremePointXMax, ExtremePointYMin,
                          ExtremePointYMax, ExtremePointZMin,
                          ExtremePointZMax});

    // now we find the most distant pair among 6 extreme points
    {
        float distance_x, distance_y, distance_z;
        distance_x = Length(*ExtremePointXMax - *ExtremePointXMin);
        distance_y = Length(*ExtremePointYMax - *ExtremePointYMin);
        distance_z = Length(*ExtremePointZMax - *ExtremePointZMin);

        int max_distance_index = (distance_x < distance_y)
                                     ? ((distance_y < distance_z) ? 2 : 1)
                                     : ((distance_x < distance_z) ? 2 : 0);

        PointPtr A, B;

        switch (max_distance_index) {
            case 0:
                A = ExtremePointXMin;
                B = ExtremePointXMax;
                break;
            case 1:
                A = ExtremePointYMin;
                B = ExtremePointYMax;
                break;
            case 2:
                A = ExtremePointZMin;
                B = ExtremePointZMax;
                break;
            default:
                assert(0);
        }

        ExtremePoints.erase(A);
        ExtremePoints.erase(B);

        // now we have 2 points on the convex hull, find the 3rd one among
        // remain extreme points
        PointPtr C;
        {
            double max_distance = 0.0f;

            for (const auto& point_ptr : ExtremePoints) {
                Vector3f numerator;
                Vector3f denominator;
                CrossProduct(numerator, *point_ptr - *A, *point_ptr - *B);
                denominator = *B - *A;
                double distance = Length(numerator) / Length(denominator);
                if (distance > max_distance) {
                    C = point_ptr;
                    max_distance = distance;
                }
            }
        }

        if (!C) return false;

        point_set.erase(A);
        point_set.erase(B);
        point_set.erase(C);

        // now we find the 4th point to form a tetrahedron
        PointPtr D;
        {
            float max_distance = 0.0f;

            for (const auto& point_ptr : point_set) {
                auto distance = PointToPlaneDistance({A, B, C}, *point_ptr);
                if (distance > max_distance) {
                    D = point_ptr;
                    max_distance = distance;
                }
            }
        }

        if (!D) return false;

        center_of_tetrahedron = make_shared<Point>((*A + *B + *C + *D) * 0.25f);
        hull.AddTetrahedron({A, B, C, D});
        point_set.erase(D);
    }

    return true;
}

bool QuickHull::Iterate(Polyhedron& hull, PointSet& point_set) {
    auto point_num_before = point_set.size();

    if (point_num_before != 0) {
        if (hull.Faces.empty()) {
            if (!Init(hull, point_set)) return false;
        }

        cerr << "Iterate Convex Hull (" << &hull
             << ") remain points count = " << point_num_before << endl;
        IterateHull(hull, point_set);
    }

    return point_set.size() < point_num_before;
}

void QuickHull::IterateHull(Polyhedron& hull, PointSet& point_set) {
    PointPtr far_point = nullptr;
    vector<FacePtr> faces;

    AssignPointsToFaces(hull, point_set, far_point, faces);

    if (point_set.empty()) return;

    if (far_point) {
        // remove all faces this point can see and
        // create new faces by connecting all vertices
        // on the border of hole to the new point
        set<Edge> edges_on_hole;
        for_each(faces.begin(), faces.end(), [&](const FacePtr& x) {
            for (const auto& edge : x->Edges) {
                Edge reverse_edge = {edge->second, edge->first};
                if (edges_on_hole.find(*edge) != edges_on_hole.end()) {
                    // this edge is shared by faces going to be removed
                    // so it is not on the border of hole, remove it
                    edges_on_hole.erase(*edge);
                } else if (edges_on_hole.find(reverse_edge) !=
                           edges_on_hole.end()) {
                    // this edge is shared by faces going to be removed
                    // so it is not on the border of hole, remove it
                    edges_on_hole.erase(reverse_edge);
                } else {
                    // temporary add it
                    edges_on_hole.insert(*edge);
                }
            }
            hull.Faces.erase(x);
        });

        // now we have edges on the hole
        // so we create new faces by connecting
        // them with the new point
        assert(edges_on_hole.size() >= 3);
        for (const auto& edge : edges_on_hole) {
            hull.AddFace({edge.first, edge.second, far_point},
                         center_of_tetrahedron);
        }

        // now the point has been proceeded
        // so remove it from the waiting queue
        point_set.erase(far_point);
    }
}

void QuickHull::AssignPointsToFaces(const Polyhedron& hull, PointSet& point_set,
                                    PointPtr& far_point, FaceList& faces) {
    float max_distance = 0.0f;
    auto it = point_set.begin();
    while (it != point_set.end()) {
        bool isInsideHull = true;
        FaceList tmp;
        for (const auto& pFace : hull.Faces) {
            float d;
            if ((d = PointToPlaneDistance(pFace->GetVertices(), **it)) > 0.0f) {
                // record all faces
                // the point can "see" in order to extrude the
                // convex hull face
                tmp.push_back(pFace);
                isInsideHull = false;
                if (d >= max_distance) {
                    far_point = *it;
                    max_distance = d;
                }
            }
        }

        if (isInsideHull) {
            // return an iterator to the element that follows the last element
            // removed (or set::end, if the last element was removed).
            it = point_set.erase(it);
        } else {
            if (far_point == *it) faces = tmp;
            it++;
        }
    }
}
