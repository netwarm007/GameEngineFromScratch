/*
 * Copyright (C) 2016 Google, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SHELL_WIN32_H
#define SHELL_WIN32_H

#include <windows.h>
#include "Shell.h"

class ShellWin32 : public Shell {
   public:
    ShellWin32(Game &game);
    ~ShellWin32();

    void run();
    void quit();

   private:
    PFN_vkGetInstanceProcAddr load_vk();
    bool can_present(VkPhysicalDevice phy, uint32_t queue_family);

    void create_window();
    VkSurfaceKHR create_surface(VkInstance instance);

    static LRESULT CALLBACK window_proc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
        ShellWin32 *shell = reinterpret_cast<ShellWin32 *>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

        // called from constructor, CreateWindowEx specifically.  But why?
        if (!shell) return DefWindowProc(hwnd, uMsg, wParam, lParam);

        return shell->handle_message(uMsg, wParam, lParam);
    }
    LRESULT handle_message(UINT msg, WPARAM wparam, LPARAM lparam);

    HINSTANCE hinstance_;
    HWND hwnd_;

    HMODULE hmodule_;
};

#endif  // SHELL_WIN32_H
