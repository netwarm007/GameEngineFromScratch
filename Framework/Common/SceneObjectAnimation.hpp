#pragma once
#include "BaseSceneObject.hpp"
#include "Animatable.hpp"
#include "SceneObjectTransform.hpp"
#include "Curve.hpp"

namespace My {
    class SceneObjectTrack : public BaseSceneObject, implements Animatable<float>
    {
        private:
            std::shared_ptr<SceneObjectTransform> m_pTransform;
            std::shared_ptr<Curve<float>> m_Time;
            std::shared_ptr<Curve<float>> m_Value;

        public:
            SceneObjectTrack() = delete;
            SceneObjectTrack(std::shared_ptr<SceneObjectTransform>& trans, 
                             std::shared_ptr<Curve<float>>& time, 
                             std::shared_ptr<Curve<float>>& value) 
                : BaseSceneObject(SceneObjectType::kSceneObjectTypeTrack),
                  m_pTransform(trans), m_Time(time), m_Value(value)
                {}
            void Update(const float time_point) final; 

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectTrack& obj);
    };

    class SceneObjectAnimationClip : public BaseSceneObject, implements Animatable<float>
    {
        private:
            int m_nIndex = 0;
            bool m_bLoop = false;
            std::vector<std::shared_ptr<SceneObjectTrack>> m_Tracks;

        public:
            SceneObjectAnimationClip() = delete;
            SceneObjectAnimationClip(int index) : BaseSceneObject(SceneObjectType::kSceneObjectTypeAnimationClip),
                m_nIndex(index)
            {}
            int GetIndex() { return m_nIndex; }
            void AddTrack(std::shared_ptr<SceneObjectTrack>& track);
            void Update(const float time_point) final; 

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectAnimationClip& obj);
    };
}