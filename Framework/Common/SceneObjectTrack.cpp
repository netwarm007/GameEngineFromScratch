#include "SceneObjectTrack.hpp"

#include <cassert>

#include "Bezier.hpp"
#include "Linear.hpp"

using namespace My;
using namespace std;

void SceneObjectTrack::Update(const float time_point) {
    if (m_pTransform) {
        auto time_curve_type = m_Time->GetCurveType();
        auto value_curve_type = m_Value->GetCurveType();

        float proportion = 0.0f;
        size_t index = 0;
        switch (time_curve_type) {
            case CurveType::kLinear:
                proportion =
                    dynamic_pointer_cast<Linear<float, float>>(m_Time)->Reverse(
                        time_point, index);
                break;
            case CurveType::kBezier:
                proportion =
                    dynamic_pointer_cast<Bezier<float, float>>(m_Time)->Reverse(
                        time_point, index);
                break;
            default:
                assert(0);
        }

        switch (value_curve_type) {
            case CurveType::kLinear:
                switch (m_kTrackType) {
                    case SceneObjectTrackType::kScalar: {
                        auto new_val =
                            dynamic_pointer_cast<Linear<float, float>>(m_Value)
                                ->Interpolate(proportion, index);
                        UpdateTransform(new_val);
                        break;
                    }
                    case SceneObjectTrackType::kVector3: {
                        auto new_val =
                            dynamic_pointer_cast<Linear<Vector3f, Vector3f>>(
                                m_Value)
                                ->Interpolate(Vector3f(proportion), index);
                        UpdateTransform(new_val);
                        break;
                    }
                    case SceneObjectTrackType::kQuoternion: {
                        auto new_val =
                            dynamic_pointer_cast<
                                Linear<Quaternion<float>, float>>(m_Value)
                                ->Interpolate(proportion, index);
                        UpdateTransform(new_val);
                        break;
                    }
                    case SceneObjectTrackType::kMatrix: {
                        auto new_val =
                            dynamic_pointer_cast<Linear<Matrix4X4f, float>>(
                                m_Value)
                                ->Interpolate(proportion, index);
                        UpdateTransform(new_val);
                        break;
                    }
                }
                break;
            case CurveType::kBezier:
                switch (m_kTrackType) {
                    case SceneObjectTrackType::kScalar: {
                        auto new_val =
                            dynamic_pointer_cast<Bezier<float, float>>(m_Value)
                                ->Interpolate(proportion, index);
                        UpdateTransform(new_val);
                        break;
                    }
                    case SceneObjectTrackType::kVector3: {
                        auto new_val =
                            dynamic_pointer_cast<Bezier<Vector3f, Vector3f>>(
                                m_Value)
                                ->Interpolate(Vector3f(proportion), index);
                        UpdateTransform(new_val);
                        break;
                    }
                    case SceneObjectTrackType::kQuoternion: {
                        auto new_val =
                            dynamic_pointer_cast<
                                Bezier<Quaternion<float>, float>>(m_Value)
                                ->Interpolate(proportion, index);
                        UpdateTransform(new_val);
                        break;
                    }
                    case SceneObjectTrackType::kMatrix: {
                        auto new_val =
                            dynamic_pointer_cast<Bezier<Matrix4X4f, float>>(
                                m_Value)
                                ->Interpolate(proportion, index);
                        UpdateTransform(new_val);
                        break;
                    }
                }
                break;
            default:
                assert(0);
        }
    }
}

template <typename U>
inline void SceneObjectTrack::UpdateTransform(const U new_val) {
    m_pTransform->Update(new_val);
}
