#include "SplashSequence.hpp"

#include "AnimationManager.hpp"
#include "IPhysicsManager.hpp"
#include "ISceneManager.hpp"

using namespace My;

void SplashSequence::RestoreNodes() {
    if (m_Text) m_Text->SetPhysicsEnabled(m_TextPhysics);
    if (m_Ball) {
        m_Ball->SetPhysicsEnabled(m_BallPhysics);
        m_Ball->SetVisibility(m_BallVisible);
    }
    m_Text.reset();
    m_Ball.reset();
    m_Waiting = false;
    m_EndTime = 0.0f;
}

void SplashSequence::Finalize() {
    RestoreNodes();
    m_Revision = 0;
}

void SplashSequence::Tick() {
    const auto revision = m_Scenes.GetSceneRevision();
    const auto scene = m_Scenes.GetSceneForRendering();
    if (revision != m_Revision) {
        RestoreNodes();
        m_Revision = revision;
        if (!scene) return;
        const auto textIt = scene->LUT_Name_GeometryNode.find("Text");
        const auto ballIt = scene->LUT_Name_GeometryNode.find("Ball");
        if (textIt == scene->LUT_Name_GeometryNode.end() ||
            ballIt == scene->LUT_Name_GeometryNode.end()) {
            return;
        }
        auto text = textIt->second.lock();
        auto ball = ballIt->second.lock();
        if (!text || !ball) return;

        BaseSceneNode::animation_clip_iterator clip;
        if (!text->GetFirstAnimationClip(clip)) return;
        do {
            if (clip->second->GetEndTime() > m_EndTime) {
                m_EndTime = clip->second->GetEndTime();
            }
        } while (text->GetNextAnimationClip(clip));
        if (m_EndTime <= 0.0f) return;

        m_Text = std::move(text);
        m_Ball = std::move(ball);
        m_TextPhysics = m_Text->PhysicsEnabled();
        m_BallPhysics = m_Ball->PhysicsEnabled();
        m_BallVisible = m_Ball->Visible();
        m_Text->SetPhysicsEnabled(false);
        m_Ball->SetPhysicsEnabled(false);
        m_Ball->SetVisibility(false);
        m_Physics.DeleteRigidBody(*m_Text);
        m_Physics.DeleteRigidBody(*m_Ball);
        m_Waiting = true;
    }

    if (!m_Waiting || m_Animations.GetTimelineTime() < m_EndTime) return;

    // Sample the exact endpoint before handing transform ownership to physics.
    BaseSceneNode::animation_clip_iterator clip;
    if (m_Text->GetFirstAnimationClip(clip)) {
        do {
            clip->second->Update(m_EndTime);
        } while (m_Text->GetNextAnimationClip(clip));
    }
    m_Text->SetPhysicsEnabled(m_TextPhysics);
    m_Ball->SetPhysicsEnabled(m_BallPhysics);
    m_Physics.CreateRigidBody(
        m_Text, *scene->GetGeometry(m_Text->GetSceneObjectRef()));
    m_Physics.CreateRigidBody(
        m_Ball, *scene->GetGeometry(m_Ball->GetSceneObjectRef()));
    m_Ball->SetVisibility(m_BallVisible);
    m_Waiting = false;
}
