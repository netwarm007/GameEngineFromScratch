#include "quickhull.hpp"

using namespace My;
using namespace std;

void QuickHull::Init()
{
    ComputeInitialTetrahydron();
}

bool QuickHull::Iterate()
{
    if (m_PointWaitProcess.size() != 0)
        IterateHull();
    return m_PointWaitProcess.size() != 0;
}

void QuickHull::ComputeInitialTetrahydron()
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
    m_PointWaitProcess = m_PointSet;

    // finding the Extreme Points [O(n) complexity]
    for(auto point_ptr : m_PointWaitProcess)
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

        m_PointWaitProcess.erase(A);
        m_PointWaitProcess.erase(B);
        m_PointWaitProcess.erase(C);

        // now we find the 4th point to form a tetrahydron
        PointPtr D;
        {
            float max_distance = 0;

            for (auto point_ptr : m_PointWaitProcess)
            {
                auto distance = PointToPlaneDistance({A,B,C}, point_ptr);
                if (distance > max_distance)
                {
                    D = point_ptr;
                    max_distance = distance;
                }
            }
        }

        center_of_tetrahydron = make_shared<Point>((*A + *B + *C + *D) * 0.25f);
        m_ConvexHull.AddTetrahydron({A,B,C,D});
        m_PointWaitProcess.erase(D);
    }
}

void QuickHull::IterateHull() 
{
    AssignPointsToFaces();

    cerr << "remain point count: " << m_PointWaitProcess.size() << endl;
    auto pPoint = *m_PointWaitProcess.begin();
    auto pFace = m_PointAboveWhichFacies.find(pPoint)->second;
    float max_distance = 0.0f;
    auto vertices = pFace->GetVertices();
    auto range = m_PointsAboveFace.equal_range(pFace);
    PointPtr far_point = nullptr;
    for_each(
                range.first,
                range.second,
                [&](decltype(m_PointsAboveFace)::value_type x)
                { 
                    auto distance = PointToPlaneDistance(vertices, x.second);
                    if (distance > max_distance)
                    {
                        far_point = x.second;
                        max_distance = distance;
                    }
                }
    );

    if (far_point)
    {
        // remove all faces this point can see and
        // create new faces by connecting all vertices
        // on the border of hole to the new point
        set<Edge> edges_on_hole;
        auto range = m_PointAboveWhichFacies.equal_range(far_point);
        for_each(
                    range.first,
                    range.second,
                    [&](decltype(m_PointAboveWhichFacies)::value_type x)
                    { 
                        auto face_to_be_removed = x.second;
                        for (auto edge : face_to_be_removed->Edges)
                        {
                            if (edges_on_hole.find(*edge) != edges_on_hole.end())
                            {
                                // this edge is shared by faces going to be removed
                                // so it is not on the border of hole, remove it
                                edges_on_hole.erase(*edge);
                            }
                            else
                            {
                                // temporary add it
                                edges_on_hole.insert(*edge);
                            }
                        }
                        m_ConvexHull.Faces.erase(face_to_be_removed); 
                    }
        );

        // now we have edges on the hole
        // so we create new faces by connecting
        // them with the new point
        //assert(edges_on_hole.size() >= 3);
        for (auto edge : edges_on_hole)
        {
            m_ConvexHull.AddFace({edge.first, edge.second, far_point}, center_of_tetrahydron);
        }

        // now the point has been proceeded
        // so remove it from the waiting queue
        m_PointWaitProcess.erase(far_point);
    }
}

void QuickHull::AssignPointsToFaces()
{
    m_PointsAboveFace.clear();
    m_PointAboveWhichFacies.clear();

    auto it = m_PointWaitProcess.begin();
    while (it != m_PointWaitProcess.end())
    {
        bool isInsideHull = true;
        for (auto pFace : m_ConvexHull.Faces)
        {
            if (isPointAbovePlane(pFace->GetVertices(), *it))
            {
                m_PointsAboveFace.insert({pFace, *it});

                // record all faces
                // the point can "see" in order to extrude the
                // convex hull face
                m_PointAboveWhichFacies.insert({*it, pFace});
                isInsideHull = false;
            }
        }

        if (isInsideHull)
        {
            // return an iterator to the element that follows the last element removed 
            // (or set::end, if the last element was removed).
            it = m_PointWaitProcess.erase(it);
        }
        else
        {
            it++;
        }
    }
}

