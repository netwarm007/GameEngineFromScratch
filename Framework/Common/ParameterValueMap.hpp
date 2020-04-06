#pragma once
#include <utility>



#include "geommath.hpp"
#include "portable.hpp"
#include "SceneObjectTexture.hpp"

namespace My {
    template <typename T>
    struct ParameterValueMap
    {
        T Value;
        std::shared_ptr<SceneObjectTexture> ValueMap;

        ParameterValueMap() = default;

        ParameterValueMap(const T value) : Value(value), ValueMap(nullptr) {};
        ParameterValueMap(std::shared_ptr<SceneObjectTexture>  value) : ValueMap(std::move(value)) {};

        ParameterValueMap(const ParameterValueMap<T>& rhs) = default;

        ParameterValueMap(ParameterValueMap<T>&& rhs) = default;

        ParameterValueMap& operator=(const ParameterValueMap<T>& rhs) = default;
        ParameterValueMap& operator=(ParameterValueMap<T>&& rhs) = default;
        ParameterValueMap& operator=(const std::shared_ptr<SceneObjectTexture>& rhs) 
        {
            ValueMap = rhs;
            return *this;
        };

        ~ParameterValueMap() = default;

        friend std::ostream& operator<<(std::ostream& out, const ParameterValueMap<T>& obj)
        {
            out << "Parameter Value: " << obj.Value << std::endl;
            if (obj.ValueMap) {
                out << "Parameter Map: " << *obj.ValueMap << std::endl;
            }

            return out;
        }
    };

    typedef ParameterValueMap<Vector4f> Color;
    typedef ParameterValueMap<Vector3f> Normal;
    typedef ParameterValueMap<float>    Parameter;
}
