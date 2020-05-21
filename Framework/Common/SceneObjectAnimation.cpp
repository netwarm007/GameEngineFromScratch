#include "SceneObjectAnimation.hpp"

using namespace My;
using namespace std;

void SceneObjectAnimationClip::AddTrack(shared_ptr<SceneObjectTrack>& track) {
    m_Tracks.push_back(track);
}

void SceneObjectAnimationClip::Update(const float time_point) {
    for (const auto& track : m_Tracks) {
        track->Update(time_point);
    }
}
