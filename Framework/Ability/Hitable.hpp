#pragma once
#include "Boundable.hpp"
#include "Intersectable.hpp"
#include "portable.hpp"

#include <map>

namespace My {
ENUM(HitableType){kNone, kGeometry, kList, kBVH};

template <class T>
class Hitable : _implements_ Intersectable<T>, _implements_ Boundable<T> {
   public:
    HitableType type = HitableType::kNone;
    static const std::map<HitableType, std::string> HitableTypeNames;
    auto GetTypeName() const { return HitableTypeNames.at(type); }

    friend std::ostream& operator<<(std::ostream& out, const Hitable<T>& obj) {
        out << "Hitable: " << obj.GetTypeName() << '\t';
        obj.dump(out);

        return out;
    }

   private:
    virtual std::ostream& dump(std::ostream& out) const = 0;
};

template<class T>
const std::map<HitableType, std::string> Hitable<T>::HitableTypeNames = {{HitableType::kNone, "None"},
                                    {HitableType::kGeometry, "Geomery"},
                                    {HitableType::kList, "List"},
                                    {HitableType::kBVH, "BVH"}};

}  // namespace My
