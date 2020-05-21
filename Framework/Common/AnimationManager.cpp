#include "AnimationManager.hpp"

#include "SceneManager.hpp"

using namespace My;
using namespace std;

int AnimationManager::Initialize() { return 0; }

void AnimationManager::Finalize() { ClearAnimationClips(); }

void AnimationManager::Tick() {
    auto rev = g_pSceneManager->GetSceneRevision();
    if (m_nSceneRevision != rev) {
        cerr << "[AnimationManager] Detected Scene Change, reinitialize "
                "animations ..."
             << endl;
        ClearAnimationClips();

        auto& scene = g_pSceneManager->GetSceneForRendering();

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

    if (!m_bTimeLineStarted) {
        m_TimeLineStartPoint = std::chrono::steady_clock::now();
        m_bTimeLineStarted = true;
    }

    m_TimeLineValue = std::chrono::steady_clock::now() - m_TimeLineStartPoint;

    for (const auto& clip : m_AnimationClips) {
        clip->Update(m_TimeLineValue.count());
    }
}

void AnimationManager::AddAnimationClip(
    const std::shared_ptr<SceneObjectAnimationClip>& clip) {
    m_AnimationClips.push_back(clip);
}

void AnimationManager::ClearAnimationClips() { m_AnimationClips.clear(); }
