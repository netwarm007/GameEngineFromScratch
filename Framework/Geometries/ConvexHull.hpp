#pragma once
#include "Polyhedron.hpp"
#include "quickhull.hpp"

namespace My {
class ConvexHull : public Polyhedron, protected QuickHull {
   public:
    ConvexHull() = default;
    ~ConvexHull() override = default;

    explicit ConvexHull(PointSet& point_set) : m_PointSet(point_set){};
    explicit ConvexHull(PointSet&& point_set)
        : m_PointSet(std::move(point_set)){};

   public:
    void AddPoint(const Point& new_point) {
        m_PointSet.insert(std::make_shared<Point>(new_point));
        m_bFullyBuild = false;
    }
    void AddPoint(const Vector3& new_point) {
        m_PointSet.insert(std::make_shared<Point>(Point(
            {(float)new_point[0], (float)new_point[1], (float)new_point[2]})));
        m_bFullyBuild = false;
    }
    void AddPoint(const PointPtr& new_point) {
        m_PointSet.insert(new_point);
        m_bFullyBuild = false;
    }
    void AddPoint(PointPtr&& new_point) {
        m_PointSet.insert(std::move(new_point));
        m_bFullyBuild = false;
    }
    void AddPointSet(const PointSet& point_set) {
        m_PointSet.insert(point_set.begin(), point_set.end());
        m_bFullyBuild = false;
    }
    bool Iterate() {
        if (!m_bFullyBuild) {
            m_bFullyBuild = !QuickHull::Iterate(*this, m_PointSet);
        }

        return !m_bFullyBuild;
    }
    [[nodiscard]] PointSet GetPointSet() const { return m_PointSet; }
    [[nodiscard]] Polyhedron GetHull() const {
        return *static_cast<const Polyhedron*>(this);
    }

   protected:
    PointSet m_PointSet;
    bool m_bFullyBuild = false;
};
}  // namespace My
