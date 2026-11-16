#pragma once
#include "Animatable.hpp"
#include "BaseSceneObject.hpp"
#include "Curve.hpp"
#include "SceneObjectTrack.hpp"
#include "SceneObjectTransform.hpp"

namespace My {
class SceneObjectAnimationClip : public BaseSceneObject,
                                 _implements_ Animatable<float> {
   private:
    int m_nIndex = 0;
    float m_EndTime{0.0f};
    std::vector<std::shared_ptr<SceneObjectTrack>> m_Tracks;

   public:
    SceneObjectAnimationClip() = delete;
    explicit SceneObjectAnimationClip(int index)
        : BaseSceneObject(SceneObjectType::kSceneObjectTypeAnimationClip),
          m_nIndex(index) {}
    int GetIndex() { return m_nIndex; }
    float GetEndTime() const { return m_EndTime; }
    void SetEndTime(float time) { m_EndTime = time; }
    void AddTrack(std::shared_ptr<SceneObjectTrack>& track);
    void Update(const float time_point) final;

    friend std::ostream& operator<<(std::ostream& out,
                                    const SceneObjectAnimationClip& obj);
};
}  // namespace My
