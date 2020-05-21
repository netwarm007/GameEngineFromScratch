#pragma once
#include <utility>

#include "SceneObjectTexture.hpp"
#include "geommath.hpp"
#include "portable.hpp"

namespace My {
template <typename T>
struct ParameterValueMap {
    T Value;
    std::shared_ptr<SceneObjectTexture> ValueMap;

    ParameterValueMap() = default;
    ~ParameterValueMap() = default;

    explicit ParameterValueMap(const T value)
        : Value(value), ValueMap(nullptr){};
    explicit ParameterValueMap(std::shared_ptr<SceneObjectTexture> value)
        : ValueMap(std::move(value)){};

    ParameterValueMap& operator=(
        const std::shared_ptr<SceneObjectTexture>& rhs) {
        ValueMap = rhs;
        return *this;
    };

    ParameterValueMap& operator=(std::shared_ptr<SceneObjectTexture>&& rhs) {
        ValueMap.swap(rhs);
        return *this;
    };

    friend std::ostream& operator<<(std::ostream& out,
                                    const ParameterValueMap<T>& obj) {
        out << "Parameter Value: " << obj.Value << std::endl;
        if (obj.ValueMap) {
            out << "Parameter Map: " << *obj.ValueMap << std::endl;
        }

        return out;
    }
};

using Color = ParameterValueMap<Vector4f>;
using Normal = ParameterValueMap<Vector3f>;
using Parameter = ParameterValueMap<float>;
}  // namespace My
