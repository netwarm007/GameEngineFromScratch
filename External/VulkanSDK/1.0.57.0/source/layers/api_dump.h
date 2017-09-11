/* Copyright (c) 2015-2016 The Khronos Group Inc.
 * Copyright (c) 2015-2016 Valve Corporation
 * Copyright (c) 2015-2016 LunarG, Inc.
 * Copyright (C) 2015-2016 Google Inc.
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
 *
 * Author: Lenny Komow <lenny@lunarg.com>
 */

#pragma once

#define NOMINMAX

#include "vk_loader_platform.h"
#include "vulkan/vk_layer.h"
#include "vk_layer_config.h"
#include "vk_layer_table.h"
#include "vk_layer_extension_utils.h"
#include "vk_layer_utils.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string.h>
#include <string>
#include <type_traits>
#include <map>
#include <unordered_map>
#include <vector>
#include <unordered_set>
#include <utility>

#define MAX_STRING_LENGTH 1024

enum class ApiDumpFormat {
    Text,
    Html,
};

class ApiDumpSettings {
   public:
    ApiDumpSettings() {
        // Get the output file settings and create a stream for it
        const char *file_option = getLayerOption("lunarg_api_dump.file");
        if (file_option != NULL && strcmp(file_option, "TRUE") == 0) {
            use_cout = false;
            const char *filename_option = getLayerOption("lunarg_api_dump.log_filename");
            if (filename_option != NULL && strcmp(filename_option, "") != 0)
                output_stream.open(filename_option, std::ofstream::out | std::ostream::trunc);
            else
                output_stream.open("vk_apidump.txt", std::ofstream::out | std::ostream::trunc);
        } else {
            use_cout = true;
        }

        // Get the remaining settings
        output_format = readFormatOption("lunarg_api_dump.output_format", ApiDumpFormat::Text);

        show_params = readBoolOption("lunarg_api_dump.detailed", true);
        show_address = !readBoolOption("lunarg_api_dump.no_addr", false);
        should_flush = readBoolOption("lunarg_api_dump.flush", true);
        indent_size = std::max(readIntOption("lunarg_api_dump.indent_size", 4), 0);
        show_type = readBoolOption("lunarg_api_dump.show_types", true);
        name_size = std::max(readIntOption("lunarg_api_dump.name_size", 32), 0);
        type_size = std::max(readIntOption("lunarg_api_dump.type_size", 0), 0);
        use_spaces = readBoolOption("lunarg_api_dump.use_spaces", true);
        show_shader = readBoolOption("lunarg_api_dump.show_shader", false);

        // Generate HTML heading if specified
        if (output_format == ApiDumpFormat::Html) {
            // Find the layer path
            std::string layer_path;
#ifdef _WIN32
            char temp[MAX_STRING_LENGTH];
            int bytes = GetEnvironmentVariableA("VK_LAYER_PATH", temp, MAX_STRING_LENGTH - 1);
            if (0 < bytes) {
                std::string location = temp;
                layer_path = location.substr(0, location.rfind("\\"));

                size_t index = 0;
                while (true) {
                    index = layer_path.find("\\", index);
                    if (index == std::string::npos) {
                        break;
                    }
                    layer_path.replace(index, 1, "/");
                    index++;
                }
            } else {
                layer_path = "";
            }
#elif __GNUC__
            layer_path = getenv("VK_LAYER_PATH");
            if (layer_path.length() > 0) {
                layer_path = layer_path.substr(0, layer_path.rfind("/"));
            } else {
                layer_path = "";
            }
#endif

            // clang-format off
            // Insert html heading
            stream() <<
                "<!doctype html>"
                "<html>"
                    "<head>"
                        "<title>Vulkan API Dump</title>"
                        "<style type='text/css'>"
                        "html {"
                            "background-color: #0b1e48;"
                            "background-image: url('https://vulkan.lunarg.com/img/bg-starfield.jpg');"
                            "background-position: center;"
                            "-webkit-background-size: cover;"
                            "-moz-background-size: cover;"
                            "-o-background-size: cover;"
                            "background-size: cover;"
                            "background-attachment: fixed;"
                            "background-repeat: no-repeat;"
                            "height: 100%;"
                        "}"
                        "#header {"
                            "z-index: -1;"
                        "}"
                        "#header>img {"
                            "position: absolute;"
                            "width: 160px;"
                            "margin-left: -280px;"
                            "top: -10px;"
                            "left: 50%;"
                        "}"
                        "#header>h1 {"
                            "font-family: Arial, 'Helvetica Neue', Helvetica, sans-serif;"
                            "font-size: 44px;"
                            "font-weight: 200;"
                            "text-shadow: 4px 4px 5px #000;"
                            "color: #eee;"
                            "position: absolute;"
                            "width: 400px;"
                            "margin-left: -80px;"
                            "top: 8px;"
                            "left: 50%;"
                        "}"
                        "body {"
                            "font-family: Consolas, monaco, monospace;"
                            "font-size: 14px;"
                            "line-height: 20px;"
                            "color: #eee;"
                            "height: 100%;"
                            "margin: 0;"
                            "overflow: hidden;"
                        "}"
                        "#wrapper {"
                            "background-color: rgba(0, 0, 0, 0.7);"
                            "border: 1px solid #446;"
                            "box-shadow: 0px 0px 10px #000;"
                            "padding: 8px 12px;"
                            "display: inline-block;"
                            "position: absolute;"
                            "top: 80px;"
                            "bottom: 25px;"
                            "left: 50px;"
                            "right: 50px;"
                            "overflow: auto;"
                        "}"
                        "details>*:not(summary) {"
                            "margin-left: 22px;"
                        "}"
                        "summary:only-child {"
                          "display: block;"
                          "padding-left: 15px;"
                        "}"
                        "details>summary:only-child::-webkit-details-marker {"
                            "display: none;"
                            "padding-left: 15px;"
                        "}"
                        ".var, .type, .val {"
                            "display: inline;"
                            "margin: 0 6px;"
                        "}"
                        ".type {"
                            "color: #acf;"
                        "}"
                        ".val {"
                            "color: #afa;"
                            "text-align: right;"
                        "}"
                        ".thd {"
                            "color: #888;"
                        "}"
                        "</style>"
                    "</head>"
                    "<body>"
                        "<div id='header'>"
                            "<img src='https://lunarg.com/wp-content/uploads/2016/02/LunarG-wReg-150.png' />"
                            "<h1>Vulkan API Dump</h1>"
                        "</div>"
                        "<div id='wrapper'>";
            // clang-format on
        }
    }

    ~ApiDumpSettings() {
        if (output_format == ApiDumpFormat::Html) {
            // Close off html
            stream() << "</div></body></html>";
        }
        if (!use_cout)
            output_stream.close();
    }

    inline ApiDumpFormat format() const { return output_format; }

    std::ostream &formatNameType(std::ostream &stream, int indents, const char *name, const char *type) const {
        stream << indentation(indents) << name << ": ";

        if (use_spaces)
            stream << spaces(name_size - (int)strlen(name) - 2);
        else
            stream << tabs((name_size - (int)strlen(name) - 3 + indent_size) / indent_size);

        if (show_type && use_spaces)
            stream << type << spaces(type_size - (int)strlen(type));
        else if (show_type && !use_spaces)
            stream << type << tabs((type_size - (int)strlen(type) - 1 + indent_size) / indent_size);

        return stream << " = ";
    }

    inline const char *indentation(int indents) const {
        if (use_spaces)
            return spaces(indents * indent_size);
        else
            return tabs(indents);
    }

    inline bool shouldFlush() const { return should_flush; }

    inline bool showAddress() const { return show_address; }

    inline bool showParams() const { return show_params; }

    inline bool showShader() const { return show_shader; }

    inline bool showType() const { return show_type; }

    inline std::ostream &stream() const { return use_cout ? std::cout : *(std::ofstream *)&output_stream; }

   private:
    inline static bool readBoolOption(const char *option, bool default_value) {
        const char *string_option = getLayerOption(option);
        if (string_option != NULL && strcmp(string_option, "TRUE") == 0)
            return true;
        else if (string_option != NULL && strcmp(string_option, "FALSE") == 0)
            return false;
        else
            return default_value;
    }

    inline static int readIntOption(const char *option, int default_value) {
        const char *string_option = getLayerOption(option);
        int value;
        if (sscanf(string_option, "%d", &value) != 1) {
            return default_value;
        } else {
            return value;
        }
    }

    inline static ApiDumpFormat readFormatOption(const char *option, ApiDumpFormat default_value) {
        const char *string_option = getLayerOption(option);
        if (strcmp(string_option, "Text") == 0)
            return ApiDumpFormat::Text;
        else if (strcmp(string_option, "Html") == 0)
            return ApiDumpFormat::Html;
        else
            return default_value;
    }

    inline static const char *spaces(int count) { return SPACES + (MAX_SPACES - std::max(count, 0)); }

    inline static const char *tabs(int count) { return TABS + (MAX_TABS - std::max(count, 0)); }

    bool use_cout;
    std::ofstream output_stream;
    ApiDumpFormat output_format;
    bool show_params;
    bool show_address;
    bool should_flush;

    bool show_type;
    int indent_size;
    int name_size;
    int type_size;
    bool use_spaces;
    bool show_shader;

    static const char *const SPACES;
    static const int MAX_SPACES = 72;
    static const char *const TABS;
    static const int MAX_TABS = 18;
};

const char *const ApiDumpSettings::SPACES = "                                                                        ";
const char *const ApiDumpSettings::TABS = "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t";

class ApiDumpInstance {
   public:
    inline ApiDumpInstance() : dump_settings(NULL), frame_count(0), thread_count(0) {
        loader_platform_thread_create_mutex(&output_mutex);
        loader_platform_thread_create_mutex(&frame_mutex);
        loader_platform_thread_create_mutex(&thread_mutex);
        loader_platform_thread_create_mutex(&cmd_buffer_state_mutex);
    }

    inline ~ApiDumpInstance() {
        if (dump_settings != NULL) delete dump_settings;

        loader_platform_thread_delete_mutex(&thread_mutex);
        loader_platform_thread_delete_mutex(&frame_mutex);
        loader_platform_thread_delete_mutex(&output_mutex);
        loader_platform_thread_delete_mutex(&cmd_buffer_state_mutex);
    }

    inline uint64_t frameCount() {
        loader_platform_thread_lock_mutex(&frame_mutex);
        uint64_t count = frame_count;
        loader_platform_thread_unlock_mutex(&frame_mutex);
        return count;
    }

    inline void nextFrame() {
        loader_platform_thread_lock_mutex(&frame_mutex);
        ++frame_count;
        loader_platform_thread_unlock_mutex(&frame_mutex);
    }

    inline loader_platform_thread_mutex *outputMutex() { return &output_mutex; }

    inline const ApiDumpSettings &settings() {
        if (dump_settings == NULL) dump_settings = new ApiDumpSettings();

        return *dump_settings;
    }

    uint32_t threadID() {
        loader_platform_thread_id id = loader_platform_get_thread_id();
        loader_platform_thread_lock_mutex(&thread_mutex);
        for (uint32_t i = 0; i < thread_count; ++i) {
            if (thread_map[i] == id) {
                loader_platform_thread_unlock_mutex(&thread_mutex);
                return i;
            }
        }

        uint32_t new_index = thread_count;
        thread_map[thread_count++] = id;
        assert(thread_count < MAX_THREADS);
        loader_platform_thread_unlock_mutex(&thread_mutex);
        return new_index;
    }

    inline VkCommandBufferLevel getCmdBufferLevel(VkCommandBuffer cmd_buffer) {
        loader_platform_thread_lock_mutex(&cmd_buffer_state_mutex);

        const auto level_iter = cmd_buffer_level.find(cmd_buffer);
        assert(level_iter != cmd_buffer_level.end());
        const auto level = level_iter->second;

        loader_platform_thread_unlock_mutex(&cmd_buffer_state_mutex);
        return level;
    }

    inline void eraseCmdBuffers(VkDevice device, VkCommandPool cmd_pool, std::vector<VkCommandBuffer> cmd_buffers) {
        cmd_buffers.erase(std::remove(cmd_buffers.begin(), cmd_buffers.end(), nullptr), cmd_buffers.end());
        if (!cmd_buffers.empty()) {
            loader_platform_thread_lock_mutex(&cmd_buffer_state_mutex);

            const auto pool_cmd_buffers_iter = cmd_buffer_pools.find(std::make_pair(device, cmd_pool));
            assert(pool_cmd_buffers_iter != cmd_buffer_pools.end());

            for (const auto cmd_buffer : cmd_buffers) {
                pool_cmd_buffers_iter->second.erase(cmd_buffer);

                assert(cmd_buffer_level.count(cmd_buffer) > 0);
                cmd_buffer_level.erase(cmd_buffer);
            }

            loader_platform_thread_unlock_mutex(&cmd_buffer_state_mutex);
        }
    }

    inline void addCmdBuffers(VkDevice device, VkCommandPool cmd_pool, std::vector<VkCommandBuffer> cmd_buffers,
                              VkCommandBufferLevel level) {
        loader_platform_thread_lock_mutex(&cmd_buffer_state_mutex);

        auto &pool_cmd_buffers = cmd_buffer_pools[std::make_pair(device, cmd_pool)];
        pool_cmd_buffers.insert(cmd_buffers.begin(), cmd_buffers.end());

        for (const auto cmd_buffer : cmd_buffers) {
            assert(cmd_buffer_level.count(cmd_buffer) == 0);
            cmd_buffer_level[cmd_buffer] = level;
        }

        loader_platform_thread_unlock_mutex(&cmd_buffer_state_mutex);
    }

    inline void eraseCmdBufferPool(VkDevice device, VkCommandPool cmd_pool) {
        if (cmd_pool != VK_NULL_HANDLE) {
            loader_platform_thread_lock_mutex(&cmd_buffer_state_mutex);

            const auto cmd_buffers_iter = cmd_buffer_pools.find(std::make_pair(device, cmd_pool));
            assert(cmd_buffers_iter != cmd_buffer_pools.end());
            cmd_buffers_iter->second.clear();

            loader_platform_thread_unlock_mutex(&cmd_buffer_state_mutex);
        }
    }

    static inline ApiDumpInstance &current() { return current_instance; }

   private:
    static ApiDumpInstance current_instance;

    ApiDumpSettings *dump_settings;
    loader_platform_thread_mutex output_mutex;
    loader_platform_thread_mutex frame_mutex;
    uint64_t frame_count;

    static const size_t MAX_THREADS = 513;
    loader_platform_thread_mutex thread_mutex;
    loader_platform_thread_id thread_map[MAX_THREADS];
    uint32_t thread_count;

    loader_platform_thread_mutex cmd_buffer_state_mutex;
    std::map<std::pair<VkDevice, VkCommandPool>, std::unordered_set<VkCommandBuffer> > cmd_buffer_pools;
    std::unordered_map<VkCommandBuffer, VkCommandBufferLevel> cmd_buffer_level;
};

ApiDumpInstance ApiDumpInstance::current_instance;

//==================================== Text Backend Helpers ======================================//

template <typename T, typename... Args>
inline void dump_text_array(const T *array, size_t len, const ApiDumpSettings &settings, const char *type_string,
                            const char *child_type, const char *name, int indents,
                            std::ostream &(*dump)(const T, const ApiDumpSettings &, int, Args... args), Args... args) {
    settings.formatNameType(settings.stream(), indents, name, type_string);
    if (array == NULL) {
        settings.stream() << "NULL\n";
        return;
    }
    if (settings.showAddress())
        settings.stream() << (void *)array << "\n";
    else
        settings.stream() << "address\n";

    for (size_t i = 0; i < len && array != NULL; ++i) {
        std::stringstream stream;
        stream << name << '[' << i << ']';
        std::string indexName = stream.str();
        dump_text_value(array[i], settings, child_type, indexName.c_str(), indents + 1, dump, args...);
    }
}

template <typename T, typename... Args>
inline void dump_text_array(const T *array, size_t len, const ApiDumpSettings &settings, const char *type_string,
                            const char *child_type, const char *name, int indents,
                            std::ostream &(*dump)(const T &, const ApiDumpSettings &, int, Args... args), Args... args) {
    settings.formatNameType(settings.stream(), indents, name, type_string);
    if (array == NULL) {
        settings.stream() << "NULL\n";
        return;
    }
    if (settings.showAddress())
        settings.stream() << (void *)array << "\n";
    else
        settings.stream() << "address\n";

    for (size_t i = 0; i < len && array != NULL; ++i) {
        std::stringstream stream;
        stream << name << '[' << i << ']';
        std::string indexName = stream.str();
        dump_text_value(array[i], settings, child_type, indexName.c_str(), indents + 1, dump, args...);
    }
}

template <typename T, typename... Args>
inline void dump_text_pointer(const T *pointer, const ApiDumpSettings &settings, const char *type_string, const char *name,
                              int indents, std::ostream &(*dump)(const T, const ApiDumpSettings &, int, Args... args),
                              Args... args) {
    if (pointer == NULL) {
        settings.formatNameType(settings.stream(), indents, name, type_string);
        settings.stream() << "NULL\n";
    } else {
        dump_text_value(*pointer, settings, type_string, name, indents, dump, args...);
    }
}

template <typename T, typename... Args>
inline void dump_text_pointer(const T *pointer, const ApiDumpSettings &settings, const char *type_string, const char *name,
                              int indents, std::ostream &(*dump)(const T &, const ApiDumpSettings &, int, Args... args),
                              Args... args) {
    if (pointer == NULL) {
        settings.formatNameType(settings.stream(), indents, name, type_string);
        settings.stream() << "NULL\n";
    } else {
        dump_text_value(*pointer, settings, type_string, name, indents, dump, args...);
    }
}

template <typename T, typename... Args>
inline void dump_text_value(const T object, const ApiDumpSettings &settings, const char *type_string, const char *name, int indents,
                            std::ostream &(*dump)(const T, const ApiDumpSettings &, int, Args... args), Args... args) {
    settings.formatNameType(settings.stream(), indents, name, type_string);
    dump(object, settings, indents, args...) << "\n";
}

template <typename T, typename... Args>
inline void dump_text_value(const T &object, const ApiDumpSettings &settings, const char *type_string, const char *name,
                            int indents, std::ostream &(*dump)(const T &, const ApiDumpSettings &, int, Args... args),
                            Args... args) {
    settings.formatNameType(settings.stream(), indents, name, type_string);
    dump(object, settings, indents, args...);
}

inline void dump_text_special(const char *text, const ApiDumpSettings &settings, const char *type_string, const char *name,
                              int indents) {
    settings.formatNameType(settings.stream(), indents, name, type_string);
    settings.stream() << text << "\n";
}

inline bool dump_text_bitmaskOption(const std::string &option, std::ostream &stream, bool isFirst) {
    if (isFirst)
        stream << " (";
    else
        stream << " | ";
    stream << option;
    return false;
}

inline std::ostream &dump_text_cstring(const char *object, const ApiDumpSettings &settings, int indents) {
    if (object == NULL)
        return settings.stream() << "NULL";
    else
        return settings.stream() << "\"" << object << "\"";
}

inline std::ostream &dump_text_void(const void *object, const ApiDumpSettings &settings, int indents) {
    if (object == NULL)
        return settings.stream() << "NULL";
    else if (settings.showAddress())
        return settings.stream() << object;
    else
        return settings.stream() << "address";
}


inline std::ostream &dump_text_int(int object, const ApiDumpSettings &settings, int indents) {
    return settings.stream() << object;
}

//==================================== Html Backend Helpers ======================================//

inline std::ostream &dump_html_nametype(std::ostream &stream, bool showType, const char *name, const char *type) {
    stream << "<div class='var'>" << name << "</div>";
    if (showType) {
        stream << "<div class='type'>" << type << "</div>";
    }
    return stream;
}

template <typename T, typename... Args>
inline void dump_html_array(const T *array, size_t len, const ApiDumpSettings &settings, const char *type_string,
                            const char *child_type, const char *name, int indents,
                            std::ostream &(*dump)(const T, const ApiDumpSettings &, int, Args... args), Args... args) {
    if (array == NULL) {
        settings.stream() << "<details class='data'><summary>";
        dump_html_nametype(settings.stream(), settings.showType(), name, type_string);
        settings.stream() << "<div class='val'>";
        settings.stream() << "NULL</div></summary></details>";
        return;
    }
    settings.stream() << "<details class='data'><summary>";
    dump_html_nametype(settings.stream(), settings.showType(), name, type_string);
    settings.stream() << "<div class='val'>";
    if (settings.showAddress())
        settings.stream() << (void *)array << "\n";
    else
        settings.stream() << "address\n";
    settings.stream() << "</div></summary>";
    for (size_t i = 0; i < len && array != NULL; ++i) {
        std::stringstream stream;
        stream << name << '[' << i << ']';
        std::string indexName = stream.str();
        dump_html_value(array[i], settings, child_type, indexName.c_str(), indents + 1, dump, args...);
    }
    settings.stream() << "</details>";
}

template <typename T, typename... Args>
inline void dump_html_array(const T *array, size_t len, const ApiDumpSettings &settings, const char *type_string,
                            const char *child_type, const char *name, int indents,
                            std::ostream &(*dump)(const T &, const ApiDumpSettings &, int, Args... args), Args... args) {
    if (array == NULL) {
        settings.stream() << "<details class='data'><summary>";
        dump_html_nametype(settings.stream(), settings.showType(), name, type_string);
        settings.stream() << "<div class='val'>";
        settings.stream() << "NULL</div></summary></details>";
        return;
    }
    settings.stream() << "<details class='data'><summary>";
    dump_html_nametype(settings.stream(), settings.showType(), name, type_string);
    settings.stream() << "<div class='val'>";
    if (settings.showAddress())
        settings.stream() << (void *)array << "\n";
    else
        settings.stream() << "address\n";
    settings.stream() << "</div></summary>";
    for (size_t i = 0; i < len && array != NULL; ++i) {
        std::stringstream stream;
        stream << name << '[' << i << ']';
        std::string indexName = stream.str();
        dump_html_value(array[i], settings, child_type, indexName.c_str(), indents + 1, dump, args...);
    }
    settings.stream() << "</details>";
}

template <typename T, typename... Args>
inline void dump_html_pointer(const T *pointer, const ApiDumpSettings &settings, const char *type_string, const char *name,
                              int indents, std::ostream &(*dump)(const T, const ApiDumpSettings &, int, Args... args),
                              Args... args) {
    if (pointer == NULL) {
        settings.stream() << "<details class='data'><summary>";
        dump_html_nametype(settings.stream(), settings.showType(), name, type_string);
        settings.stream() << "<div class='val'>NULL</div></summary></details>";
    } else {
        dump_html_value(*pointer, settings, type_string, name, indents, dump, args...);
    }
}

template <typename T, typename... Args>
inline void dump_html_pointer(const T *pointer, const ApiDumpSettings &settings, const char *type_string, const char *name,
                              int indents, std::ostream &(*dump)(const T &, const ApiDumpSettings &, int, Args... args),
                              Args... args) {
    if (pointer == NULL) {
        settings.stream() << "<details class='data'><summary>";
        dump_html_nametype(settings.stream(), settings.showType(), name, type_string);
        settings.stream() << "<div class='val'>NULL</div></summary></details>";
    } else {
        dump_html_value(*pointer, settings, type_string, name, indents, dump, args...);
    }
}

template <typename T, typename... Args>
inline void dump_html_value(const T object, const ApiDumpSettings &settings, const char *type_string, const char *name, int indents,
                            std::ostream &(*dump)(const T, const ApiDumpSettings &, int, Args... args), Args... args) {
    settings.stream() << "<details class='data'><summary>";
    dump_html_nametype(settings.stream(), settings.showType(), name, type_string);
    dump(object, settings, indents, args...);
    settings.stream() << "</details>";
}

template <typename T, typename... Args>
inline void dump_html_value(const T &object, const ApiDumpSettings &settings, const char *type_string, const char *name,
                            int indents, std::ostream &(*dump)(const T &, const ApiDumpSettings &, int, Args... args),
                            Args... args) {
    settings.stream() << "<details class='data'><summary>";
    dump_html_nametype(settings.stream(), settings.showType(), name, type_string);
    dump(object, settings, indents, args...);
    settings.stream() << "</details>";
}

inline void dump_html_special(
    const char *text, const ApiDumpSettings &settings, const char *type_string,
    const char *name, int indents) {
    settings.stream() << "<details class='data'><summary>";
    dump_html_nametype(settings.stream(), settings.showType(), name, type_string);
    settings.stream() << "<div class='val'>" << text << "</div></summary></details>";
}

inline bool dump_html_bitmaskOption(const std::string &option,
                                    std::ostream &stream, bool isFirst) {
    if (isFirst)
        stream << " (";
    else
        stream << " | ";
    stream << option;
    return false;
}

inline std::ostream &dump_html_cstring(const char *object,
                                       const ApiDumpSettings &settings,
                                       int indents) {
    settings.stream() << "<div class='val'>";
    if (object == NULL)
        settings.stream() << "NULL";
    else
        settings.stream() << "\"" << object << "\"";
    return settings.stream() << "</div>";
}

inline std::ostream &dump_html_void(const void *object,
                                    const ApiDumpSettings &settings,
                                    int indents) {
    settings.stream() << "<div class='val'>";
    if (object == NULL)
        settings.stream() << "NULL";
    else if (settings.showAddress())
        settings.stream() << object;
    else
        settings.stream() << "address";
    return settings.stream() << "</div>";
}

inline std::ostream &dump_html_int(int object, const ApiDumpSettings &settings, int indents) {
    settings.stream() << "<div class='val'>";
    settings.stream() << object;
    return settings.stream() << "</div>";
}
