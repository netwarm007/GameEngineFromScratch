#pragma once

#include <cassert>
#include <iostream>

#include "imgui.h"

namespace My {
class GuiContext {
   public:
    GuiContext() = default;
    GuiContext(const GuiContext&) = delete;
    GuiContext& operator=(const GuiContext&) = delete;
    ~GuiContext() { Finalize(); }

    int Initialize() {
        if (m_Context || ImGui::GetCurrentContext()) {
            std::cerr << "GUI context initialization requires no active context\n";
            return -1;
        }
        IMGUI_CHECKVERSION();
        m_Context = ImGui::CreateContext();
        return 0;
    }

    void Finalize() {
        if (!m_Context) return;
        assert(ImGui::GetCurrentContext() == m_Context);
        assert(!ImGui::GetIO().BackendPlatformUserData &&
               !ImGui::GetIO().BackendRendererUserData &&
               "GUI backends must shut down before their context");
        ImGui::DestroyContext(m_Context);
        m_Context = nullptr;
    }

   private:
    ImGuiContext* m_Context = nullptr;
};
}  // namespace My
