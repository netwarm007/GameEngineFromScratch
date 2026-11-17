#include "FrameAllocator.hpp"

using namespace My;

FrameAllocator::FrameAllocator(IMemoryManager* pMmgr, size_t capacityPerFrame,
                               size_t frameCount)
    : IAllocator(pMmgr) {
    Initialize(capacityPerFrame, frameCount);
}

FrameAllocator::~FrameAllocator() { Release(); }

bool FrameAllocator::Initialize(size_t capacityPerFrame, size_t frameCount) {
    Release();
    if (capacityPerFrame == 0 || frameCount < 2) return false;

    std::vector<std::unique_ptr<LinearAllocator>> frames;
    frames.reserve(frameCount);
    for (size_t i = 0; i < frameCount; i++) {
        auto frame =
            std::make_unique<LinearAllocator>(m_pMemoryManager, capacityPerFrame);
        if (!frame->IsConfigured()) return false;
        frames.push_back(std::move(frame));
    }

    m_Frames = std::move(frames);
    return true;
}

void FrameAllocator::Release() {
    m_Frames.clear();
    m_nCurrent = 0;
    m_nFrameNumber = 0;
}

void FrameAllocator::BeginFrame() {
    if (m_Frames.empty()) return;
    m_nCurrent = (m_nCurrent + 1) % m_Frames.size();
    m_Frames[m_nCurrent]->FreeAll();
    ++m_nFrameNumber;
}

void* FrameAllocator::Allocate(size_t size, size_t alignment) {
    if (m_Frames.empty()) return nullptr;
    return m_Frames[m_nCurrent]->Allocate(size, alignment);
}

void FrameAllocator::FreeAll() {
    for (auto& frame : m_Frames) frame->FreeAll();
}

size_t FrameAllocator::CapacityPerFrame() const {
    return m_Frames.empty() ? 0 : m_Frames[m_nCurrent]->Capacity();
}

size_t FrameAllocator::Used() const {
    return m_Frames.empty() ? 0 : m_Frames[m_nCurrent]->Used();
}

size_t FrameAllocator::Remaining() const {
    return m_Frames.empty() ? 0 : m_Frames[m_nCurrent]->Remaining();
}

size_t FrameAllocator::FailedAllocationCount() const {
    return m_Frames.empty() ? 0 : m_Frames[m_nCurrent]->FailedAllocationCount();
}
