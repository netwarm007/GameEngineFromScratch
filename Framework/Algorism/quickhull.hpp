#pragma once
#include <cassert>
#include <limits>
#include <memory>
#include <set>
#include <utility>
#include "geommath.hpp"

namespace My {
    class QuickHull {
    public:
        QuickHull() = default;
        ~QuickHull() = default;
        QuickHull(const PointSet& point_set) : m_PointSet(point_set) {}
        QuickHull(PointSet&& point_set) : m_PointSet(std::move(point_set)) {}
        void AddPoint(const PointPtr& new_point) { m_PointSet.insert(new_point); }
        void AddPoint(PointPtr&& new_point) { m_PointSet.insert(std::move(new_point)); }
        void AddPointSet(const PointSet& point_set) { m_PointSet.insert(point_set.begin(), point_set.end()); }
        void ComputeHull() { ComputeHullInternal(); }
        const PointSet& GetHull() const { return m_ConvexHull; }

    protected:
        void ComputeHullInternal() 
        {
            if(m_PointSet.size() < 4)
            {
                // too few points in the point set, nothing could be done
                return;
            }

            PointPtr ExtremePointXMin = std::make_shared<Point>(std::numeric_limits<float>::max());
            PointPtr ExtremePointYMin = std::make_shared<Point>(std::numeric_limits<float>::max());
            PointPtr ExtremePointZMin = std::make_shared<Point>(std::numeric_limits<float>::max());
            PointPtr ExtremePointXMax = std::make_shared<Point>(std::numeric_limits<float>::min());
            PointPtr ExtremePointYMax = std::make_shared<Point>(std::numeric_limits<float>::min());
            PointPtr ExtremePointZMax = std::make_shared<Point>(std::numeric_limits<float>::min());

            // copy the point set into temporary working set
            PointSet workingPointSet = m_PointSet;

            // finding the Extreme Points [O(n) complexity]
            for(auto point_ptr : workingPointSet)
            {
                if(point_ptr->x < ExtremePointXMin->x)
                    ExtremePointXMin = point_ptr;
                if(point_ptr->y < ExtremePointYMin->y)
                    ExtremePointYMin = point_ptr;
                if(point_ptr->z < ExtremePointZMin->z)
                    ExtremePointZMin = point_ptr;
                if(point_ptr->x > ExtremePointXMax->x)
                    ExtremePointXMax = point_ptr;
                if(point_ptr->y > ExtremePointYMax->y)
                    ExtremePointYMax = point_ptr;
                if(point_ptr->z > ExtremePointZMax->z)
                    ExtremePointZMax = point_ptr;
            }

            PointSet ExtremePoints;
            ExtremePoints.insert({
                                    ExtremePointXMin, ExtremePointXMax,
                                    ExtremePointYMin, ExtremePointYMax,
                                    ExtremePointZMin, ExtremePointZMax
                                });

            // now we find the most distant pair among 6 extreme points
            {
                float distance_x, distance_y, distance_z;
                distance_x = Length(*ExtremePointXMax - *ExtremePointXMin);
                distance_y = Length(*ExtremePointYMax - *ExtremePointYMin);
                distance_z = Length(*ExtremePointZMax - *ExtremePointZMin);

                int max_distance_index = (distance_x < distance_y)?
                                                (
                                                    (distance_y < distance_z)?2:1
                                                )
                                                :
                                                (
                                                    (distance_x < distance_z)?2:0
                                                );

                PointPtr A, B;

                switch(max_distance_index)
                {
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

                // now we have 2 points on the convex hull, find the 3rd one among remain 
                // extreme points
                PointPtr C;
                {
                    double max_distance = 0.0f;

                    for (auto point_ptr : ExtremePoints)
                    {
                        Vector3f numerator;
                        Vector3f denominator;
                        CrossProduct(numerator, *point_ptr - *A, *point_ptr - *B);
                        denominator = *B - *A;
                        double distance = Length(numerator) / Length(denominator);
                        if (distance > max_distance)
                        {
                            C = point_ptr;
                            max_distance = distance;
                        }
                    }
                }

                m_ConvexHull.insert({A, B, C});
                workingPointSet.erase(A);
                workingPointSet.erase(B);
                workingPointSet.erase(C);
            }

        }

    protected:
        PointSet m_PointSet;
        PointSet m_ConvexHull;
    };
}