#include "quickhull.hpp"

using namespace My;
using namespace std;

void QuickHull::ComputeHullInternal() 
{
    if(m_PointSet.size() < 4)
    {
        // too few points in the point set, nothing could be done
        return;
    }

    PointPtr ExtremePointXMin = make_shared<Point>(numeric_limits<float>::max());
    PointPtr ExtremePointYMin = make_shared<Point>(numeric_limits<float>::max());
    PointPtr ExtremePointZMin = make_shared<Point>(numeric_limits<float>::max());
    PointPtr ExtremePointXMax = make_shared<Point>(numeric_limits<float>::min());
    PointPtr ExtremePointYMax = make_shared<Point>(numeric_limits<float>::min());
    PointPtr ExtremePointZMax = make_shared<Point>(numeric_limits<float>::min());

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

        workingPointSet.erase(A);
        workingPointSet.erase(B);
        workingPointSet.erase(C);

        // now we find the 4th point to form a tetrahydron
        PointPtr D;
        {
            float max_distance = 0.0f;

            for (auto point_ptr : workingPointSet)
            {
                Vector3f normal;
                float distance;
                CrossProduct(normal, *B - *A, *C - *A);
                Normalize(normal);
                DotProduct(distance, normal, *point_ptr - *A);
                if (distance > max_distance)
                {
                    D = point_ptr;
                    max_distance = distance;
                }
            }
        }

        m_ConvexHull.AddTetrahydron({A,B,C,D});
        workingPointSet.erase(D);
    }

}
