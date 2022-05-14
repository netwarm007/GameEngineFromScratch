// dear imgui: Platform Binding for IOS / UIKit 

#include "imgui.h"      // IMGUI_IMPL_API

@class NSEvent;
@class NSView;

IMGUI_IMPL_API bool     ImGui_ImplIOS_Init();
IMGUI_IMPL_API void     ImGui_ImplIOS_Shutdown();
IMGUI_IMPL_API bool     ImGui_ImplIOS_HandleEvent(NSEvent *_Nonnull event, NSView *_Nullable view);
