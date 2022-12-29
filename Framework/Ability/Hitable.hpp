#pragma once
#include "Boundable.hpp"
#include "Intersectable.hpp"
#include "portable.hpp"

namespace My {
ENUM(HitableType){kNone, kGeometry, kList, kBVH};

template <class T>
class Hitable : _implements_ Intersectable<T>, _implements_ Boundable<T> {
   public:
    HitableType type = HitableType::kNone;

    friend std::ostream& operator<<(std::ostream& out, const Hitable<T>& obj) {
        out << "Hitable: "; 
        switch(obj.type) {
            case HitableType::kNone: out << "None"; break;
            case HitableType::kGeometry: out << "Geometry"; break;
            case HitableType::kList: out << "List"; break;
            case HitableType::kBVH: out << "BVN"; break;
        }
        out << '\t';
        obj.dump(out);

        return out;
    }

   private:
    virtual std::ostream& dump(std::ostream& out) const = 0;
};

}  // namespace My
