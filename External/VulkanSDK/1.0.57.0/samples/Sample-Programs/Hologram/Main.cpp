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

#include <string>
#include <vector>

#include "Hologram.h"

namespace {

Game *create_game(const std::vector<std::string> &args) { return new Hologram(args); }

Game *create_game(int argc, char **argv) {
    std::vector<std::string> args(argv, argv + argc);
    return create_game(args);
}

}  // namespace

#if defined(VK_USE_PLATFORM_XCB_KHR)

#include "ShellXcb.h"

int main(int argc, char **argv) {
    Game *game = create_game(argc, argv);
    {
        ShellXcb shell(*game);
        shell.run();
    }
    delete game;

    return 0;
}

#elif defined(VK_USE_PLATFORM_ANDROID_KHR)

#include <android/log.h>
#include "ShellAndroid.h"

void android_main(android_app *app) {
    Game *game = create_game(ShellAndroid::get_args(*app));
    try {
        ShellAndroid shell(*app, *game);
        shell.run();
    } catch (const std::runtime_error &e) {
        __android_log_print(ANDROID_LOG_ERROR, game->settings().name.c_str(), "%s", e.what());
    }

    delete game;
}

#elif defined(VK_USE_PLATFORM_WIN32_KHR)

#include "ShellWin32.h"

int main(int argc, char **argv) {
    Game *game = create_game(argc, argv);
    {
        ShellWin32 shell(*game);
        shell.run();
    }
    delete game;

    return 0;
}

#endif  // VK_USE_PLATFORM_XCB_KHR
