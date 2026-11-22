#include "RuntimeRoot.hpp"

#include <iostream>

#include "EngineAllocation.hpp"

using namespace My;

RuntimeRoot::RuntimeRoot() = default;
RuntimeRoot::~RuntimeRoot() { ReleaseMemorySystem(); }

bool RuntimeRoot::InitializeMemorySystem(const MemorySystem::Config& config) {
    if (m_pMemorySystem) return m_pMemorySystem->IsConfigured();
    auto system = std::make_unique<MemorySystem>(config);
    if (!system->IsConfigured()) {
        std::cerr << "Engine allocators unavailable; using default heap\n";
        return false;
    }
    m_pMemorySystem = std::move(system);
    SetEngineAllocator(&m_pMemorySystem->Allocator());
    return true;
}

void RuntimeRoot::ReleaseMemorySystem() {
    if (!m_pMemorySystem) return;
    SetEngineAllocator(nullptr);
    m_pMemorySystem.reset();
}