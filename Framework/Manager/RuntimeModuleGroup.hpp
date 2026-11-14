#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "IRuntimeModule.hpp"

namespace My {
// Modules are borrowed and must outlive the group. Register before Initialize.
class RuntimeModuleGroup {
   public:
    void Register(IRuntimeModule& module) {
        if (m_Initialized) {
            throw std::logic_error("Cannot register a running module group");
        }
        if (std::find(m_Modules.begin(), m_Modules.end(), &module) !=
            m_Modules.end()) {
            throw std::invalid_argument("Module is already registered");
        }
        m_Modules.push_back(&module);
    }

    int Initialize() {
        if (m_Initialized) {
            std::cerr << "Module group is already initialized\n";
            return -1;
        }
        for (auto* module : m_Modules) {
            const int result = module->Initialize();
            if (result != 0) {
                std::cerr << "Module initialize failed at index "
                          << m_InitializedCount << " (error " << result << ")\n";
                Unwind();
                return result;
            }
            ++m_InitializedCount;
        }
        m_Initialized = true;
        return 0;
    }

    void Finalize() {
        Unwind();
        m_Modules.clear();
        m_Initialized = false;
    }

    void Tick() {
        for (size_t i = 0; i < m_InitializedCount; ++i) {
            m_Modules[i]->Tick();
        }
    }

   private:
    void Unwind() {
        while (m_InitializedCount != 0) {
            m_Modules[--m_InitializedCount]->Finalize();
        }
    }

    std::vector<IRuntimeModule*> m_Modules;
    size_t m_InitializedCount = 0;
    bool m_Initialized = false;
};
}  // namespace My
