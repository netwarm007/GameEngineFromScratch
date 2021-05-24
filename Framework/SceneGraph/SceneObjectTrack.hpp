#pragma once
#include <utility>

#include "Animatable.hpp"
#include "BaseSceneObject.hpp"
#include "Curve.hpp"
#include "SceneObjectTransform.hpp"
#include "geommath.hpp"
#include "portable.hpp"

namespace My {
ENUM(SceneObjectTrackType){kScalar = "SCAL"_i32, kVector3 = "VEC3"_i32,
                           kQuoternion = "QUON"_i32, kMatrix = "MATX"_i32};

class SceneObjectTrack : public BaseSceneObject,
                         _implements_ Animatable<float> {
   private:
    std::shared_ptr<SceneObjectTransform> m_pTransform;
    std::shared_ptr<CurveBase> m_Time;
    std::shared_ptr<CurveBase> m_Value;
    const SceneObjectTrackType m_kTrackType;

   public:
    SceneObjectTrack() = delete;
    SceneObjectTrack(std::shared_ptr<SceneObjectTransform> trans,
                     std::shared_ptr<CurveBase> time,
                     std::shared_ptr<CurveBase> value,
                     SceneObjectTrackType type)
        : BaseSceneObject(SceneObjectType::kSceneObjectTypeTrack),
          m_pTransform(std::move(trans)),
          m_Time(std::move(time)),
          m_Value(std::move(value)),
          m_kTrackType(type) {}
    void Update(const float time_point) final;

   private:
    template <typename U>
    void UpdateTransform(const U new_val);

    friend std::ostream& operator<<(std::ostream& out,
                                    const SceneObjectTrack& obj);
};

}  // namespace My