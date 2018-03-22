#pragma once
#include <cassert>
#include <limits>
#include <unordered_map>
#include <memory>
#include <utility>
#include "geommath.hpp"
#include "Polyhedron.hpp"

namespace My {
    class QuickHull {
    public:
        QuickHull() = default;
        ~QuickHull() = default;
        bool Iterate(Polyhedron& hull, PointSet& point_set);

    protected:
        bool Init(Polyhedron& hull, PointSet& point_set);
        void IterateHull(Polyhedron& hull, PointSet& point_set); 
        void AssignPointsToFaces(const Polyhedron& hull, PointSet& point_set);

    protected:
        std::unordered_multimap<FacePtr, PointPtr> m_PointsAboveFace;
        std::unordered_multimap<PointPtr, FacePtr> m_PointAboveWhichFacies;

        PointPtr center_of_tetrahedron;
    };
}