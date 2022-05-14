// dear imgui: Platform Binding for iOS / UIKit

#include "imgui.h"
#include "imgui_impl_ios.h"

#import <UIKit/UIKit.h>


// Data

// Functions
bool ImGui_ImplIOS_Init()
{
    ImGuiIO& io = ImGui::GetIO();

    // Setup back-end capabilities flags
    io.BackendPlatformName = "imgui_impl_ios";

    // Keyboard mapping. Dear ImGui will use those indices to peek into the io.KeyDown[] array.
    const int offset_for_function_keys = 256 - 0xF700;
    io.KeyMap[ImGuiKey_Tab]             = '\t';
    io.KeyMap[ImGuiKey_Backspace]       = 127;
    io.KeyMap[ImGuiKey_Space]           = 32;
    io.KeyMap[ImGuiKey_Enter]           = 13;
    io.KeyMap[ImGuiKey_Escape]          = 27;
    io.KeyMap[ImGuiKey_KeyPadEnter]     = 13;
    io.KeyMap[ImGuiKey_A]               = 'A';
    io.KeyMap[ImGuiKey_C]               = 'C';
    io.KeyMap[ImGuiKey_V]               = 'V';
    io.KeyMap[ImGuiKey_X]               = 'X';
    io.KeyMap[ImGuiKey_Y]               = 'Y';
    io.KeyMap[ImGuiKey_Z]               = 'Z';

    return true;
}

void ImGui_ImplIOS_Shutdown()
{
}

static int mapCharacterToKey(int c)
{
    if (c >= 'a' && c <= 'z')
        return c - 'a' + 'A';
    if (c == 25) // SHIFT+TAB -> TAB
        return 9;
    if (c >= 0 && c < 256)
        return c;
    if (c >= 0xF700 && c < 0xF700 + 256)
        return c - 0xF700 + 256;
    return -1;
}

static void resetKeys()
{
    ImGuiIO& io = ImGui::GetIO();
    for (int n = 0; n < IM_ARRAYSIZE(io.KeysDown); n++)
        io.KeysDown[n] = false;
}

bool ImGui_ImplIOS_HandleEvent(NSEvent* event, NSView* view)
{
    ImGuiIO& io = ImGui::GetIO();

    return false;
}
