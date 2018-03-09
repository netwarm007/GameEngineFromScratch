#pragma once
#include <cassert>
#include <limits>
#include <unordered_map>
#include <memory>
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
        const PointSet& GetPointSet() const { return m_PointSet; }
        void Init();
        bool Iterate();
        const Polyhedron& GetHull() const { return m_ConvexHull; }

    protected:
        void ComputeInitialTetrahydron();
        void IterateHull();
        void AssignPointsToFaces();

    protected:
        PointSet m_PointSet;
        Polyhedron m_ConvexHull;

        // temporary buffers
        PointSet m_PointWaitProcess;
        std::unordered_multimap<FacePtr, PointPtr> m_PointsAboveFace;
        std::unordered_multimap<PointPtr, FacePtr> m_PointAboveWhichFacies;

    private:
        PointPtr center_of_tetrahydron;
    };
}