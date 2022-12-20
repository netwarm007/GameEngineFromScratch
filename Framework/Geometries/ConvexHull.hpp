#pragma once
#include "Polyhedron.hpp"
#include "quickhull.hpp"

namespace My {
template <typename T>
class ConvexHull : public Polyhedron<T>, protected QuickHull<T> {
   public:
    ConvexHull() = default;
    ~ConvexHull() override = default;

    explicit ConvexHull(PointSet<T>& point_set) : m_PointSet(point_set){};
    explicit ConvexHull(PointSet<T>&& point_set)
        : m_PointSet(std::move(point_set)){};

   public:
    void AddPoint(const Point<T>& new_point) {
        m_PointSet.insert(std::make_shared<Point<T>>(new_point));
        m_bFullyBuild = false;
    }
    void AddPoint(const PointPtr<T>& new_point) {
        m_PointSet.insert(new_point);
        m_bFullyBuild = false;
    }
    void AddPoint(PointPtr<T>&& new_point) {
        m_PointSet.insert(std::move(new_point));
        m_bFullyBuild = false;
    }
    void AddPointSet(const PointSet<T>& point_set) {
        m_PointSet.insert(point_set.begin(), point_set.end());
        m_bFullyBuild = false;
    }
    bool Iterate() {
        if (!m_bFullyBuild) {
            m_bFullyBuild = !QuickHull<T>::Iterate(*this, m_PointSet);
        }

        return !m_bFullyBuild;
    }
    [[nodiscard]] PointSet<T> GetPointSet() const { return m_PointSet; }
    [[nodiscard]] Polyhedron<T> GetHull() const {
        return *static_cast<const Polyhedron<T>*>(this);
    }

   protected:
    PointSet<T> m_PointSet;
    bool m_bFullyBuild = false;
};
}  // namespace My
