#include "AnimationManager.hpp"

using namespace My;
using namespace std;

int AnimationManager::Initialize() { return 0; }

void AnimationManager::Finalize() { ClearAnimationClips(); }

void AnimationManager::Tick() {
    SynchronizeScene();
    if (!m_bTimeLineStarted) {
        m_TimeLineStartPoint = std::chrono::steady_clock::now();
        m_bTimeLineStarted = true;
    }

    const std::chrono::duration<float> elapsed =
        std::chrono::steady_clock::now() - m_TimeLineStartPoint;
    Update(elapsed.count());
}

void AnimationManager::SynchronizeScene() {
    auto rev = m_Scenes.GetSceneRevision();
    if (m_nSceneRevision != rev) {
        cerr << "[AnimationManager] Detected Scene Change, reinitialize "
                "animations ..."
             << endl;
        ClearAnimationClips();

        auto& scene = m_Scenes.GetSceneForRendering();

        for (const auto& node : scene->AnimatableNodes) {
            auto pNode = node.lock();
            if (pNode) {
                BaseSceneNode::animation_clip_iterator it;
                if (pNode->GetFirstAnimationClip(it)) {
                    do {
                        AddAnimationClip(it->second);
                    } while (pNode->GetNextAnimationClip(it));
                }
            }
        }

        m_nSceneRevision = rev;

        // reset timeline
        m_bTimeLineStarted = false;
    }
}

void AnimationManager::Update(float timePoint) {
    SynchronizeScene();
    m_TimeLineValue = std::chrono::duration<float>(timePoint);
    for (const auto& clip : m_AnimationClips) {
        clip->Update(m_TimeLineValue.count());
    }
}

void AnimationManager::AddAnimationClip(
    const std::shared_ptr<SceneObjectAnimationClip>& clip) {
    m_AnimationClips.push_back(clip);
}

void AnimationManager::ClearAnimationClips() { m_AnimationClips.clear(); }
