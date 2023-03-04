#pragma once
#include "Polyhedron.hpp"
#include "geommath.hpp"

namespace My {
template <typename T>
class QuickHull {
   public:
    QuickHull() = default;
    ~QuickHull() = default;
    bool Iterate(Polyhedron<T>& hull, PointSet<T>& point_set) {
        auto point_num_before = point_set.size();

        if (point_num_before != 0) {
            if (hull.Faces.empty()) {
                if (!Init(hull, point_set)) return false;
            }

            std::cerr << "Iterate Convex Hull (" << &hull
                 << ") remain points count = " << point_num_before << std::endl;
            IterateHull(hull, point_set);
        }

        return point_set.size() < point_num_before;
    }

   protected:
    bool Init(Polyhedron<T>& hull, PointSet<T>& point_set) {
        if (point_set.size() < 4) {
            // too few points in the point set, nothing could be done
            return false;
        }

        PointPtr<T> ExtremePointXMin =
            std::make_shared<Point<T>>((std::numeric_limits<T>::max)()); // Windows: Work around of warning C4003: not enough arguments for function-like macro invocation 'max'
        PointPtr<T> ExtremePointYMin =
            std::make_shared<Point<T>>((std::numeric_limits<T>::max)());
        PointPtr<T> ExtremePointZMin =
            std::make_shared<Point<T>>((std::numeric_limits<T>::max)());
        PointPtr<T> ExtremePointXMax =
            std::make_shared<Point<T>>(std::numeric_limits<T>::lowest());
        PointPtr<T> ExtremePointYMax =
            std::make_shared<Point<T>>(std::numeric_limits<T>::lowest());
        PointPtr<T> ExtremePointZMax =
            std::make_shared<Point<T>>(std::numeric_limits<T>::lowest());

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

        PointSet<T> ExtremePoints;
        ExtremePoints.insert({ExtremePointXMin, ExtremePointXMax,
                              ExtremePointYMin, ExtremePointYMax,
                              ExtremePointZMin, ExtremePointZMax});

        // now we find the most distant pair among 6 extreme points
        {
            float distance_x, distance_y, distance_z;
            distance_x = Length(*ExtremePointXMax - *ExtremePointXMin);
            distance_y = Length(*ExtremePointYMax - *ExtremePointYMin);
            distance_z = Length(*ExtremePointZMax - *ExtremePointZMin);

            int max_distance_index = (distance_x < distance_y)
                                         ? ((distance_y < distance_z) ? 2 : 1)
                                         : ((distance_x < distance_z) ? 2 : 0);

            PointPtr<T> A, B;

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
            PointPtr<T> C;
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
            PointPtr<T> D;
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

            center_of_tetrahedron =
                std::make_shared<Point<T>>((*A + *B + *C + *D) * 0.25f);
            hull.AddTetrahedron({A, B, C, D});
            point_set.erase(D);
        }

        return true;
    }

    void IterateHull(Polyhedron<T>& hull, PointSet<T>& point_set) {
        PointPtr<T> far_point = nullptr;
        std::vector<FacePtr<T>> faces;

        AssignPointsToFaces(hull, point_set, far_point, faces);

        if (point_set.empty()) return;

        if (far_point) {
            // remove all faces this point can see and
            // create new faces by connecting all vertices
            // on the border of hole to the new point
            std::set<Edge<T>> edges_on_hole;
            for_each(faces.begin(), faces.end(), [&](const FacePtr<T>& x) {
                for (const auto& edge : x->Edges) {
                    Edge<T> reverse_edge = {edge->second, edge->first};
                    if (edges_on_hole.contains(*edge)) {
                        // this edge is shared by faces going to be removed
                        // so it is not on the border of hole, remove it
                        edges_on_hole.erase(*edge);
                    } else if (edges_on_hole.contains(reverse_edge)) {
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

    static void AssignPointsToFaces(const Polyhedron<T>& hull,
                                    PointSet<T>& point_set,
                                    PointPtr<T>& far_point,
                                    FaceList<T>& faces) {
        float max_distance = 0.0f;
        auto it = point_set.begin();
        while (it != point_set.end()) {
            bool isInsideHull = true;
            FaceList<T> tmp;
            for (const auto& pFace : hull.Faces) {
                float d;
                if ((d = PointToPlaneDistance(pFace->GetVertices(), **it)) >
                    0.0f) {
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
                // return an iterator to the element that follows the last
                // element removed (or set::end, if the last element was
                // removed).
                it = point_set.erase(it);
            } else {
                if (far_point == *it) faces = tmp;
                it++;
            }
        }
    }

   protected:
    PointPtr<T> center_of_tetrahedron;
};
}  // namespace My
