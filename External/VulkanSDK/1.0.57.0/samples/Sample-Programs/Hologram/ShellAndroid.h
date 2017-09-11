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

#ifndef SHELL_ANDROID_H
#define SHELL_ANDROID_H

#include <string>
#include <vector>

#include <android_native_app_glue.h>
#include "Shell.h"

class ShellAndroid : public Shell {
   public:
    static std::vector<std::string> get_args(android_app &app);

    ShellAndroid(android_app &app, Game &game);
    ~ShellAndroid();

    void log(LogPriority priority, const char *msg) const;

    void run();
    void quit();

   private:
    PFN_vkGetInstanceProcAddr load_vk();
    bool can_present(VkPhysicalDevice phy, uint32_t queue_family) { return true; }

    VkSurfaceKHR create_surface(VkInstance instance);

    void on_app_cmd(int32_t cmd);
    int32_t on_input_event(const AInputEvent *event);

    static inline void on_app_cmd(android_app *app, int32_t cmd);
    static inline int32_t on_input_event(android_app *app, AInputEvent *event);

    android_app &app_;

    void *lib_handle_;
};

void ShellAndroid::on_app_cmd(android_app *app, int32_t cmd) {
    auto android = reinterpret_cast<ShellAndroid *>(app->userData);
    android->on_app_cmd(cmd);
}

int32_t ShellAndroid::on_input_event(android_app *app, AInputEvent *event) {
    auto android = reinterpret_cast<ShellAndroid *>(app->userData);
    return android->on_input_event(event);
}

#endif  // SHELL_ANDROID_H
