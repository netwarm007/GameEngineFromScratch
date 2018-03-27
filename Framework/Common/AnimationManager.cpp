#include "AnimationManager.hpp"
#include "SceneManager.hpp"

using namespace My;
using namespace std;

int AnimationManager::Initialize()
{
    auto& scene = g_pSceneManager->GetSceneForRendering();

    for (auto node : scene.AnimatableNodes)
    {
        auto pNode = node.lock();
        if (pNode) {
            BaseSceneNode::animation_clip_iterator it;
            if (pNode->GetFirstAnimationClip(it))
            {
                do {
                    AddAnimationClip(it->second);
                } while (pNode->GetNextAnimationClip(it));
            }
        }
    }

    return 0;
}

void AnimationManager::Finalize()
{
    
}

void AnimationManager::Tick()
{
    if (!m_bTimeLineStarted)
    {
        m_TimeLineStartPoint = m_Clock.now();
        m_bTimeLineStarted = true;
    }

    m_TimeLineValue = m_Clock.now() - m_TimeLineStartPoint;

    for (auto clip : m_AnimationClips)
    {
        clip->Update(m_TimeLineValue.count());
    }
}

void AnimationManager::AddAnimationClip(std::shared_ptr<SceneObjectAnimationClip> clip)
{
    m_AnimationClips.push_back(clip);
}
