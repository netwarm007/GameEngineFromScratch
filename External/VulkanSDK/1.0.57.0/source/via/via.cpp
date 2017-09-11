/*
 * Copyright (c) 2016-2017 Valve Corporation
 * Copyright (c) 2016-2017 LunarG, Inc.
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
 * Author: Mark Young <marky@lunarg.com>
 */

#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <time.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

const char APP_VERSION[] = "Version 1.1";
#define MAX_STRING_LENGTH 1024

#ifdef _WIN32
#pragma warning(disable : 4996)
#include "shlwapi.h"
#else
#include <stdlib.h>
#include <sys/types.h>
#include <sys/statvfs.h>
#include <sys/utsname.h>
#include <dirent.h>
#include <unistd.h>
#include <dlfcn.h>
#endif

#include <json/json.h>

#include <vulkan/vulkan.h>

#if (defined(_MSC_VER) && _MSC_VER < 1900 /*vs2015*/) || defined MINGW_HAS_SECURE_API
#include <basetsd.h>
#define snprintf sprintf_s
#endif

enum ElementAlign { ALIGN_LEFT = 0, ALIGN_CENTER, ALIGN_RIGHT };

struct PhysicalDeviceInfo {
    VkPhysicalDevice vulkan_phys_dev;
    std::vector<VkQueueFamilyProperties> queue_fam_props;
};

struct GlobalItems {
    std::ofstream html_file_stream;
    bool sdk_found;
    std::string sdk_path;
    VkInstance instance;
    std::vector<PhysicalDeviceInfo> phys_devices;
    std::vector<VkDevice> log_devices;
    uint32_t cur_table;
    std::string exe_directory;
    bool is_odd_row;

#ifdef _WIN32
    bool is_wow64;
#endif
};

// Create a global variable used to store the global settings
GlobalItems global_items = {};

// Error messages
enum ErrorResults {
    SUCCESSFUL = 0,

    UNKNOWN_ERROR = -1,
    SYSTEM_CALL_FAILURE = -2,

    MISSING_DRIVER_REGISTRY = -20,
    MISSING_DRIVER_JSON = -21,
    DRIVER_JSON_PARSING_ERROR = -22,
    MISSING_DRIVER_LIB = -23,
    MISSING_LAYER_JSON = -24,
    LAYER_JSON_PARSING_ERROR = -25,
    MISSING_LAYER_LIB = -26,

    VULKAN_CANT_FIND_RUNTIME = -40,
    VULKAN_CANT_FIND_DRIVER = -41,
    VULKAN_CANT_FIND_EXTENSIONS = -42,
    VULKAN_FAILED_CREATE_INSTANCE = -43,
    VULKAN_FAILED_CREATE_DEVICE = -44,
    VULKAN_FAILED_OUT_OF_MEM = -45,

    TEST_FAILED = -60,
};

// Structure used to store name/value pairs read from the
// Vulkan layer settings file (if one exists).
struct SettingPair {
    std::string name;
    std::string value;
};

void StartOutput(std::string title);
void EndOutput();
ErrorResults PrintSystemInfo(void);
ErrorResults PrintVulkanInfo(void);
ErrorResults PrintDriverInfo(void);
ErrorResults PrintRunTimeInfo(void);
ErrorResults PrintSDKInfo(void);
void PrintExplicitLayerJsonInfo(const char *layer_json_filename, Json::Value root, uint32_t num_cols);
void PrintImplicitLayerJsonInfo(const char *layer_json_filename, Json::Value root);
ErrorResults PrintLayerInfo(void);
ErrorResults PrintLayerSettingsFileInfo(void);
ErrorResults PrintTestResults(void);
std::string TrimWhitespace(const std::string &str, const std::string &whitespace = " \t\n\r");

int main(int argc, char **argv) {
    int err_val = 0;
    time_t time_raw_format;
    struct tm *ptr_time;
    char html_file_name[MAX_STRING_LENGTH];
    char full_file[MAX_STRING_LENGTH];
    char temp[MAX_STRING_LENGTH];
    const char *output_path = NULL;
    bool generate_unique_file = false;
    ErrorResults res = SUCCESSFUL;
    size_t file_name_offset = 0;
#ifdef _WIN32
    int bytes;
#elif __GNUC__
    ssize_t len;
#endif

    // Check and handle command-line arguments
    if (argc > 1) {
        for (int iii = 1; iii < argc; iii++) {
            if (0 == strcmp("--unique_output", argv[iii])) {
                generate_unique_file = true;
            } else if (0 == strcmp("--output_path", argv[iii]) && argc > (iii + 1)) {
                output_path = argv[iii + 1];
                ++iii;
            } else {
                std::cout << "Usage of via.exe:" << std::endl
                          << "    via.exe [--unique_output] "
                             "[--output_path <path>]"
                          << std::endl
                          << "          [--unique_output] Optional "
                             "parameter to generate a unique html"
                          << std::endl
                          << "                            "
                             "output file in the form "
                             "\'via_YYYY_MM_DD_HH_MM.html\'"
                          << std::endl
                          << "          [--output_path <path>"
                             "] Optional parameter to generate the output at"
                          << std::endl
                          << "                               "
                             "  a given path"
                          << std::endl;
                return -1;
            }
        }
    }

    // If the user wants a specific output path, write it to the buffer
    // and then continue writing the rest of the name below
    if (output_path != NULL) {
        file_name_offset = strlen(output_path) + 1;
        strncpy(html_file_name, output_path, MAX_STRING_LENGTH - 1);
#ifdef _WIN32
        strncpy(html_file_name + file_name_offset - 1, "\\", MAX_STRING_LENGTH - file_name_offset);
#else
        strncpy(html_file_name + file_name_offset - 1, "/", MAX_STRING_LENGTH - file_name_offset);
#endif
    }

    // If the user wants a unique file, generate a file with the current
    // time and date incorporated into it.
    if (generate_unique_file) {
        time(&time_raw_format);
        ptr_time = localtime(&time_raw_format);
        if (strftime(html_file_name + file_name_offset, MAX_STRING_LENGTH - 1, "via_%Y_%m_%d_%H_%M.html", ptr_time) == 0) {
            std::cerr << "Couldn't prepare formatted string" << std::endl;
            goto out;
        }
    } else {
        strncpy(html_file_name + file_name_offset, "via.html", MAX_STRING_LENGTH - 1 - file_name_offset);
    }

    // Write the output file to the current executing directory, or, if
    // that fails, write it out to the user's home folder.
    global_items.html_file_stream.open(html_file_name);
    if (global_items.html_file_stream.fail()) {
#ifdef _WIN32
        char home_drive[32];
        if (0 != GetEnvironmentVariableA("HOMEDRIVE", home_drive, 31) ||
            0 != GetEnvironmentVariableA("HOMEPATH", temp, MAX_STRING_LENGTH - 1)) {
            std::cerr << "Error failed to get either HOMEDRIVE or HOMEPATH "
                         "from environment settings!"
                      << std::endl;
            goto out;
        }
        snprintf(full_file, MAX_STRING_LENGTH - 1, "%s%s\\%s", home_drive, temp, html_file_name);
#else
        snprintf(full_file, MAX_STRING_LENGTH - 1, "~/%s", html_file_name);
#endif
        global_items.html_file_stream.open(full_file);
        if (global_items.html_file_stream.fail()) {
            std::cerr << "Error failed opening html file stream to "
                         "either current"
                         " folder as "
                      << html_file_name << " or home folder as " << full_file << std::endl;
            goto out;
        }
    }

    global_items.cur_table = 0;

// Determine where we are executing at.
#ifdef _WIN32
    bytes = GetModuleFileName(NULL, temp, MAX_STRING_LENGTH - 1);
    if (0 < bytes) {
        std::string exe_location = temp;
        global_items.exe_directory = exe_location.substr(0, exe_location.rfind("\\"));

        size_t index = 0;
        while (true) {
            index = global_items.exe_directory.find("\\", index);
            if (index == std::string::npos) {
                break;
            }
            global_items.exe_directory.replace(index, 1, "/");
            index++;
        }
    } else {
        global_items.exe_directory = "";
    }

#elif __GNUC__
    len = ::readlink("/proc/self/exe", temp, MAX_STRING_LENGTH - 1);
    if (0 < len) {
        std::string exe_location = temp;
        global_items.exe_directory = exe_location.substr(0, exe_location.rfind("/"));
    } else {
        global_items.exe_directory = "";
    }
#endif

    StartOutput("LunarG VIA");

    res = PrintSystemInfo();
    if (res != SUCCESSFUL) {
        goto out;
    }
    res = PrintVulkanInfo();
    if (res != SUCCESSFUL) {
        goto out;
    }
    res = PrintTestResults();
    EndOutput();

out:

    // Print out a useful message for any common errors.
    switch (res) {
        case SUCCESSFUL:
            std::cout << "SUCCESS: Validation completed properly." << std::endl;
            break;
        case SYSTEM_CALL_FAILURE:
            std::cout << "ERROR: Failure occurred during system call." << std::endl;
            break;
        case MISSING_DRIVER_REGISTRY:
            std::cout << "ERROR: Failed to find Vulkan Driver JSON in registry." << std::endl;
            break;
        case MISSING_DRIVER_JSON:
            std::cout << "ERROR: Failed to find Vulkan Driver JSON." << std::endl;
            break;
        case DRIVER_JSON_PARSING_ERROR:
            std::cout << "ERROR: Failed to properly parse Vulkan Driver JSON." << std::endl;
            break;
        case MISSING_DRIVER_LIB:
            std::cout << "ERROR: Failed to find Vulkan Driver Lib." << std::endl;
            break;
        case MISSING_LAYER_JSON:
            std::cout << "ERROR: Failed to find Vulkan Layer JSON." << std::endl;
            break;
        case LAYER_JSON_PARSING_ERROR:
            std::cout << "ERROR: Failed to properly parse Vulkan Layer JSON." << std::endl;
            break;
        case MISSING_LAYER_LIB:
            std::cout << "ERROR: Failed to find Vulkan Layer Lib." << std::endl;
            break;
        case VULKAN_CANT_FIND_RUNTIME:
            std::cout << "ERROR: Vulkan failed to find a Vulkan Runtime to use." << std::endl;
            break;
        case VULKAN_CANT_FIND_DRIVER:
            std::cout << "ERROR: Vulkan failed to find a compatible driver." << std::endl;
            break;
        case VULKAN_CANT_FIND_EXTENSIONS:
            std::cout << "ERROR: Failed to find expected Vulkan Extensions."
                      << "  This may indicate a bad driver install." << std::endl;
            break;
        case VULKAN_FAILED_CREATE_INSTANCE:
            std::cout << "ERROR: Unknown error while attempting to create Vulkan Instance." << std::endl;
            break;
        case VULKAN_FAILED_CREATE_DEVICE:
            std::cout << "ERROR: Unknown error while attempting to create Vulkan Device." << std::endl;
            break;
        case VULKAN_FAILED_OUT_OF_MEM:
            std::cout << "ERROR: Vulkan Loader, Layer, or Driver ran out of memory." << std::endl;
            break;
        case TEST_FAILED:
            std::cout << "ERROR: Unknown Test failure occurred." << std::endl;
            break;
        case UNKNOWN_ERROR:
        default:
            std::cout << "ERROR: Uknown failure occurred.  Refer to HTML for "
                         "more info"
                      << std::endl;
            break;
    }
    err_val = static_cast<int>(res);

    global_items.html_file_stream.close();

    return err_val;
}

// Output helper functions:
//=============================

// Start writing to the HTML file by creating the appropriate
// header information including the appropriate CSS and JavaScript
// items.
void StartOutput(std::string output) {
    global_items.html_file_stream << "<!DOCTYPE html>" << std::endl;
    global_items.html_file_stream << "<HTML lang=\"en\" xml:lang=\"en\" "
                                     "xmlns=\"http://www.w3.org/1999/xhtml\">"
                                  << std::endl;
    global_items.html_file_stream << std::endl << "<HEAD>" << std::endl << "    <TITLE>" << output << "</TITLE>" << std::endl;

    global_items.html_file_stream << "    <META charset=\"UTF-8\">" << std::endl
                                  << "    <style media=\"screen\" type=\"text/css\">" << std::endl
                                  << "        html {" << std::endl
                                  // By defining the color first, this won't override the background image
                                  // (unless the images aren't there).
                                  << "            background-color: #0b1e48;" << std::endl
                                  // The following changes try to load the text image twice (locally, then
                                  // off the web) followed by the background image twice (locally, then
                                  // off the web).  The background color will only show if both background
                                  // image loads fail.  In this way, a user will see their local copy on
                                  // their machine, while a person they share it with will see the web
                                  // images (or the background color).
                                  << "            background-image: url(\"file:///" << global_items.exe_directory
                                  << "/images/lunarg_via.png\"), "
                                  << "url(\"https://vulkan.lunarg.com/img/lunarg_via.png\"), "
                                     "url(\"file:///"
                                  << global_items.exe_directory << "/images/bg-starfield.jpg\"), "
                                  << "url(\"https://vulkan.lunarg.com/img/bg-starfield.jpg\");" << std::endl
                                  << "            background-position: center top, center top, center, "
                                     "center;"
                                  << std::endl
                                  << "            -webkit-background-size: auto, auto, cover, cover;" << std::endl
                                  << "            -moz-background-size: auto, auto, cover, cover;" << std::endl
                                  << "            -o-background-size: auto, auto, cover, cover;" << std::endl
                                  << "            background-size: auto, auto, cover, cover;" << std::endl
                                  << "            background-attachment: scroll, scroll, fixed, fixed;" << std::endl
                                  << "            background-repeat: no-repeat, no-repeat, no-repeat, "
                                     "no-repeat;"
                                  << std::endl
                                  << "        }" << std::endl
                                  // h1.section is used for section headers, and h1.version is used to
                                  // print out the application version text (which shows up just under
                                  // the title).
                                  << "        h1.section {" << std::endl
                                  << "            font-family: sans-serif;" << std::endl
                                  << "            font-size: 35px;" << std::endl
                                  << "            color: #FFFFFF;" << std::endl
                                  << "        }" << std::endl
                                  << "        h1.version {" << std::endl
                                  << "            font-family: sans-serif;" << std::endl
                                  << "            font-size: 25px;" << std::endl
                                  << "            color: #FFFFFF;" << std::endl
                                  << "        }" << std::endl
                                  << "        h2.note {" << std::endl
                                  << "            font-family: sans-serif;" << std::endl
                                  << "            font-size: 22px;" << std::endl
                                  << "            color: #FFFFFF;" << std::endl
                                  << "        }" << std::endl
                                  << "        table {" << std::endl
                                  << "            min-width: 600px;" << std::endl
                                  << "            width: 70%;" << std::endl
                                  << "            border-collapse: collapse;" << std::endl
                                  << "            border-color: grey;" << std::endl
                                  << "            font-family: sans-serif;" << std::endl
                                  << "        }" << std::endl
                                  << "        td.header {" << std::endl
                                  << "            padding: 18px;" << std::endl
                                  << "            border: 1px solid #ccc;" << std::endl
                                  << "            font-size: 18px;" << std::endl
                                  << "            color: #fff;" << std::endl
                                  << "        }" << std::endl
                                  << "        td.odd {" << std::endl
                                  << "            padding: 10px;" << std::endl
                                  << "            border: 1px solid #ccc;" << std::endl
                                  << "            font-size: 16px;" << std::endl
                                  << "            color: rgb(255, 255, 255);" << std::endl
                                  << "        }" << std::endl
                                  << "        td.even {" << std::endl
                                  << "            padding: 10px;" << std::endl
                                  << "            border: 1px solid #ccc;" << std::endl
                                  << "            font-size: 16px;" << std::endl
                                  << "            color: rgb(220, 220, 220);" << std::endl
                                  << "        }" << std::endl
                                  << "        tr.header {" << std::endl
                                  << "            background-color: rgba(255,255,255,0.5);" << std::endl
                                  << "        }" << std::endl
                                  << "        tr.odd {" << std::endl
                                  << "            background-color: rgba(0,0,0,0.6);" << std::endl
                                  << "        }" << std::endl
                                  << "        tr.even {" << std::endl
                                  << "            background-color: rgba(0,0,0,0.7);" << std::endl
                                  << "        }" << std::endl
                                  << "    </style>" << std::endl
                                  << "    <script src=\"https://ajax.googleapis.com/ajax/libs/jquery/"
                                  << "2.2.4/jquery.min.js\"></script>" << std::endl
                                  << "    <script type=\"text/javascript\">" << std::endl
                                  << "        $( document ).ready(function() {" << std::endl
                                  << "            $('table tr:not(.header)').hide();" << std::endl
                                  << "            $('.header').click(function() {" << std::endl
                                  << "                "
                                     "$(this).nextUntil('tr.header').slideToggle(300);"
                                  << std::endl
                                  << "            });" << std::endl
                                  << "        });" << std::endl
                                  << "    </script>" << std::endl
                                  << "</HEAD>" << std::endl
                                  << std::endl
                                  << "<BODY>" << std::endl
                                  << std::endl;
    // We need space from the top for the VIA texture
    for (uint32_t space = 0; space < 15; space++) {
        global_items.html_file_stream << "    <BR />" << std::endl;
    }
    // All the silly "&nbsp;" are to make sure the version lines up directly
    // under the  VIA portion of the log.
    global_items.html_file_stream << "    <H1 class=\"version\"><center>";
    for (uint32_t space = 0; space < 65; space++) {
        global_items.html_file_stream << "&nbsp;";
    }
    global_items.html_file_stream << APP_VERSION << "</center></h1>" << std::endl << "    <BR />" << std::endl;

    global_items.html_file_stream << "<center><h2 class=\"note\">< NOTE: Click on section name to expand "
                                     "table ></h2></center>"
                                  << std::endl
                                  << "    <BR />" << std::endl;
}

// Close out writing to the HTML file.
void EndOutput() { global_items.html_file_stream << "</BODY>" << std::endl << std::endl << "</HTML>" << std::endl; }

void BeginSection(std::string section_str) {
    global_items.html_file_stream << "    <H1 class=\"section\"><center>" << section_str << "</center></h1>" << std::endl;
}

void EndSection() { global_items.html_file_stream << "    <BR/>" << std::endl << "    <BR/>" << std::endl; }

void PrintStandardText(std::string section) {
    global_items.html_file_stream << "    <H2><font color=\"White\">" << section << "</font></H2>" << std::endl;
}

void PrintBeginTable(const char *table_name, uint32_t num_cols) {
    global_items.html_file_stream << "    <table align=\"center\">" << std::endl
                                  << "        <tr class=\"header\">" << std::endl
                                  << "            <td colspan=\"" << num_cols << "\" class=\"header\">" << table_name << "</td>"
                                  << std::endl
                                  << "         </tr>" << std::endl;

    global_items.is_odd_row = true;
}

void PrintBeginTableRow() {
    std::string class_str = "";
    if (global_items.is_odd_row) {
        class_str = " class=\"odd\"";
    } else {
        class_str = " class=\"even\"";
    }
    global_items.html_file_stream << "        <tr" << class_str << ">" << std::endl;
}

void PrintTableElement(std::string element, ElementAlign align = ALIGN_LEFT) {
    std::string align_str = "";
    std::string class_str = "";
    if (align == ALIGN_RIGHT) {
        align_str = " align=\"right\"";
    } else if (align == ALIGN_CENTER) {
        align_str = " align=\"center\"";
    }
    if (global_items.is_odd_row) {
        class_str = " class=\"odd\"";
    } else {
        class_str = " class=\"even\"";
    }
    global_items.html_file_stream << "            <td" << align_str << class_str << ">" << element << "</td>" << std::endl;
}

void PrintEndTableRow() {
    global_items.html_file_stream << "        </tr>" << std::endl;
    global_items.is_odd_row = !global_items.is_odd_row;
}

void PrintEndTable() { global_items.html_file_stream << "    </table>" << std::endl; }

// Generate the full library location for a file based on the location of
// the JSON file referencing it, and the library location contained in that
// JSON file.
bool GenerateLibraryPath(const char *json_location, const char *library_info, const uint32_t max_length, char *library_location) {
    bool success = false;
    char final_path[MAX_STRING_LENGTH];
    char *working_string_ptr;
    uint32_t len = (max_length > MAX_STRING_LENGTH) ? MAX_STRING_LENGTH : max_length;

    if (NULL == json_location || NULL == library_info || NULL == library_location) {
        goto out;
    }

    // Remove json file from json path to get just the file base location
    strncpy(final_path, json_location, len);
    working_string_ptr = strrchr(final_path, '\\');
    if (working_string_ptr == NULL) {
        working_string_ptr = strrchr(final_path, '/');
    }
    if (working_string_ptr != NULL) {
        working_string_ptr++;
        *working_string_ptr = '\0';
    }

    // Determine if the library is relative or absolute
    if (library_info[0] == '\\' || library_info[0] == '/' || library_info[1] == ':') {
        // Absolute path
        strncpy(library_location, library_info, len);
        success = true;
    } else {
        uint32_t i = 0;
        // Relative path, so we need to use the JSON's location
        while (library_info[i] == '.' && library_info[i + 1] == '.' &&
               (library_info[i + 2] == '\\' || library_info[i + 2] == '/')) {
            i += 3;
            // Go up a folder in the json path
            working_string_ptr = strrchr(final_path, '\\');
            if (working_string_ptr == NULL) {
                working_string_ptr = strrchr(final_path, '/');
            }
            if (working_string_ptr != NULL) {
                working_string_ptr++;
                *working_string_ptr = '\0';
            }
        }
        while (library_info[i] == '.' && (library_info[i + 1] == '\\' || library_info[i + 1] == '/')) {
            i += 2;
        }
        strncpy(library_location, final_path, MAX_STRING_LENGTH - 1);
        strncat(library_location, &library_info[i], len);
        success = true;
    }

out:
    return success;
}

#ifdef _WIN32
// Registry utility fuctions to simplify reading data from the
// Windows registry.

const char g_uninstall_reg_path[] = "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall";

bool ReadRegKeyString(HKEY regFolder, const char *keyPath, const char *valueName, const int maxLength, char *retString) {
    bool retVal = false;
    DWORD bufLen = maxLength;
    DWORD keyFlags = KEY_READ;
    HKEY hKey;
    LONG lret;

    if (global_items.is_wow64) {
        keyFlags |= KEY_WOW64_64KEY;
    }

    *retString = '\0';
    lret = RegOpenKeyExA(regFolder, keyPath, 0, keyFlags, &hKey);
    if (lret == ERROR_SUCCESS) {
        lret = RegQueryValueExA(hKey, valueName, NULL, NULL, (BYTE *)retString, &bufLen);
        if (lret == ERROR_SUCCESS) {
            retVal = true;
        }
        RegCloseKey(hKey);
    }

    return retVal;
}

bool WriteRegKeyString(HKEY regFolder, const char *keyPath, char *valueName, char *valueValue) {
    bool retVal = false;
    DWORD keyFlags = KEY_WRITE;
    HKEY hKey;
    LONG lret;

    if (global_items.is_wow64) {
        keyFlags |= KEY_WOW64_64KEY;
    }

    lret = RegOpenKeyExA(regFolder, keyPath, 0, keyFlags, &hKey);
    if (lret == ERROR_SUCCESS) {
        lret = RegSetKeyValueA(hKey, NULL, valueName, REG_SZ, (BYTE *)valueValue, (DWORD)(strlen(valueValue)));
        if (lret == ERROR_SUCCESS) {
            retVal = true;
        }
        RegCloseKey(hKey);
    }

    return retVal;
}

bool DeleteRegKeyString(HKEY regFolder, const char *keyPath, char *valueName) {
    bool retVal = false;
    DWORD keyFlags = KEY_WRITE;
    HKEY hKey;
    LONG lret;

    if (global_items.is_wow64) {
        keyFlags |= KEY_WOW64_64KEY;
    }

    lret = RegOpenKeyExA(regFolder, keyPath, 0, keyFlags, &hKey);
    if (lret == ERROR_SUCCESS) {
        lret = RegDeleteKeyValueA(hKey, NULL, valueName);
        if (lret == ERROR_SUCCESS) {
            retVal = true;
        }
        RegCloseKey(hKey);
    }

    return retVal;
}

bool ReadRegKeyDword(HKEY regFolder, const char *keyPath, const char *valueName, unsigned int *returnInt) {
    bool retVal = false;
    DWORD bufLen = sizeof(DWORD);
    DWORD keyFlags = KEY_READ;
    HKEY hKey;
    LONG lret;

    if (global_items.is_wow64) {
        keyFlags |= KEY_WOW64_64KEY;
    }

    *returnInt = 0;
    lret = RegOpenKeyExA(regFolder, keyPath, 0, keyFlags, &hKey);
    if (lret == ERROR_SUCCESS) {
        lret = RegQueryValueExA(hKey, valueName, NULL, NULL, (BYTE *)returnInt, &bufLen);
        if (lret == ERROR_SUCCESS) {
            retVal = true;
        }
        RegCloseKey(hKey);
    }

    return retVal;
}

bool FindNextRegKey(HKEY regFolder, const char *keyPath, const char *keySearch, const int itemIndex, const int maxLength,
                    char *retString) {
    bool retVal = false;
    DWORD bufLen = MAX_STRING_LENGTH - 1;
    DWORD keyFlags = KEY_ENUMERATE_SUB_KEYS | KEY_QUERY_VALUE;
    HKEY hKey;
    LONG lret;
    int itemCount = 0;

    if (global_items.is_wow64) {
        keyFlags |= KEY_WOW64_64KEY;
    }

    *retString = '\0';
    lret = RegOpenKeyExA(regFolder, keyPath, 0, keyFlags, &hKey);
    if (lret == ERROR_SUCCESS) {
        DWORD index = 0;
        char keyName[MAX_STRING_LENGTH];

        do {
            lret = RegEnumKeyExA(hKey, index, keyName, &bufLen, NULL, NULL, NULL, NULL);
            if (ERROR_SUCCESS != lret) {
                break;
            }
            if (strlen(keySearch) == 0 || NULL != strstr(keyName, keySearch)) {
                if (itemIndex == itemCount) {
                    strncpy_s(retString, maxLength, keyName, bufLen);
                    retVal = true;
                    break;
                } else {
                    itemCount++;
                }
            }
            bufLen = MAX_STRING_LENGTH - 1;
            ++index;
        } while (true);
    }

    return retVal;
}

bool FindNextRegValue(HKEY regFolder, const char *keyPath, const char *valueSearch, const int startIndex, const int maxLength,
                      char *retString, uint32_t *retValue) {
    bool retVal = false;
    DWORD bufLen = MAX_STRING_LENGTH - 1;
    DWORD keyFlags = KEY_ENUMERATE_SUB_KEYS | KEY_QUERY_VALUE;
    HKEY hKey = 0;
    LONG lret;

    if (global_items.is_wow64) {
        keyFlags |= KEY_WOW64_64KEY;
    }

    *retValue = 0;
    *retString = '\0';
    lret = RegOpenKeyExA(regFolder, keyPath, 0, keyFlags, &hKey);
    if (lret == ERROR_SUCCESS) {
        DWORD index = startIndex;
        char valueName[MAX_STRING_LENGTH];

        do {
            DWORD type = REG_DWORD;
            DWORD value = 0;
            DWORD len = 4;
            valueName[0] = '\0';

            lret = RegEnumValueA(hKey, index, valueName, &bufLen, NULL, &type, (LPBYTE)&value, &len);
            if (ERROR_SUCCESS != lret) {
                break;
            }
            if (type == REG_DWORD) {
                *retValue = value;
            }
            if (strlen(valueSearch) == 0 || NULL != strstr(valueName, valueSearch)) {
                strncpy_s(retString, maxLength, valueName, bufLen);
                retVal = true;
                break;
            }

            bufLen = MAX_STRING_LENGTH - 1;
            ++index;
        } while (true);
    }

    return retVal;
}

// Registry prototypes for Windows
bool ReadRegKeyDword(HKEY regFolder, const char *keyPath, const char *valueName, unsigned int *returnInt);
bool ReadRegKeyString(HKEY regFolder, const char *keyPath, const char *valueName, const int maxLength, char *retString);
bool FindNextRegKey(HKEY regFolder, const char *keyPath, const char *keySearch, const int startIndex, const int maxLength,
                    char *retString);
bool FindNextRegValue(HKEY regFolder, const char *keyPath, const char *valueSearch, const int startIndex, const int maxLength,
                      char *retString, uint32_t *retValue);
bool WriteRegKeyString(HKEY regFolder, const char *keyPath, char *valueName, char *valueValue);
bool DeleteRegKeyString(HKEY regFolder, const char *keyPath, char *valueName);

// Functionality to determine if this 32-bit process is running on Windows 64.
//
void IsWow64() {
    typedef BOOL(WINAPI * LPFN_ISWOW64PROCESS)(HANDLE, PBOOL);

    // IsWow64Process is not available on all supported versions of Windows.
    // Use GetModuleHandle to get a handle to the DLL that contains the function
    // and GetProcAddress to get a pointer to the function if available.

    LPFN_ISWOW64PROCESS fnIsWow64Process = (LPFN_ISWOW64PROCESS)GetProcAddress(GetModuleHandle(TEXT("kernel32")), "IsWow64Process");

    if (NULL != fnIsWow64Process) {
        BOOL isWOW = FALSE;
        if (!fnIsWow64Process(GetCurrentProcess(), &isWOW)) {
            printf("Error : Failed to determine properly if on Win64!");
        }

        if (isWOW == TRUE) {
            global_items.is_wow64 = true;
        }
    }
}

// Run the test in the specified directory with the corresponding
// command-line arguments.
// Returns 0 on no error, 1 if test file wasn't found, and -1
// on any other errors.
int RunTestInDirectory(std::string path, std::string test, std::string cmd_line) {
    int err_code = -1;
    char orig_dir[MAX_STRING_LENGTH];
    orig_dir[0] = '\0';
    if (0 != GetCurrentDirectoryA(MAX_STRING_LENGTH - 1, orig_dir) && TRUE == SetCurrentDirectoryA(path.c_str())) {
        if (TRUE == PathFileExists(test.c_str())) {
            err_code = system(cmd_line.c_str());
        } else {
            // Path to specific exe doesn't exist
            err_code = 1;
        }
        SetCurrentDirectoryA(orig_dir);
    } else {
        // Path to test doesn't exist.
        err_code = 1;
    }
    return err_code;
}

// Print out any information about the current system that we can
// capture to ease in debugging/investigation at a later time.
ErrorResults PrintSystemInfo(void) {
    ErrorResults res = SUCCESSFUL;
    OSVERSIONINFOEX os_info;
    SYSTEM_INFO sys_info;
    MEMORYSTATUSEX mem_stat;
    DWORD ser_ver = 0;
    DWORD sect_per_cluster = 0;
    DWORD bytes_per_sect = 0;
    DWORD num_free_cluster = 0;
    DWORD total_num_cluster = 0;
    char system_root_dir[MAX_STRING_LENGTH];
    char generic_string[MAX_STRING_LENGTH];
    char output_string[MAX_STRING_LENGTH];
    char os_size[32];
    std::string cur_directory;
    std::string exe_directory;

    // Determine if this 32-bit process is on Win64.
    IsWow64();

#if _WIN64
    strncpy(os_size, " 64-bit", 31);
#else
    // If WOW64 support is present, then it's a 64-bit Windows
    if (global_items.is_wow64) {
        strncpy(os_size, " 64-bit", 31);
    } else {
        strncpy(os_size, " 32-bit", 31);
    }
#endif

    BeginSection("System Info");

    // Environment section has information about the OS and the
    // execution environment.
    PrintBeginTable("Environment", 3);

    ZeroMemory(&sys_info, sizeof(SYSTEM_INFO));
    GetSystemInfo(&sys_info);

    ZeroMemory(&os_info, sizeof(OSVERSIONINFOEX));
    os_info.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);

    ZeroMemory(&mem_stat, sizeof(MEMORYSTATUSEX));
    mem_stat.dwLength = sizeof(MEMORYSTATUSEX);

    // Since this is Windows #ifdef code, determine the version of Windows
    // that the applciation is running on.  It's not trivial and has to
    // refer to items queried in the above structures as well as the
    // Windows registry.
    if (TRUE == GetVersionEx((LPOSVERSIONINFO)(&os_info))) {
        switch (os_info.dwMajorVersion) {
            case 10:
                if (os_info.wProductType == VER_NT_WORKSTATION) {
                    if (ReadRegKeyString(HKEY_LOCAL_MACHINE, "Software\\Microsoft\\Windows NT\\CurrentVersion", "ProductName",
                                         MAX_STRING_LENGTH - 1, generic_string)) {
                        PrintBeginTableRow();
                        PrintTableElement("Windows");
                        PrintTableElement(generic_string);
                        PrintTableElement(os_size);
                        PrintEndTableRow();

                        if (ReadRegKeyString(HKEY_LOCAL_MACHINE, "Software\\Microsoft\\Windows NT\\CurrentVersion", "CurrentBuild",
                                             MAX_STRING_LENGTH - 1, output_string)) {
                            PrintBeginTableRow();
                            PrintTableElement("");
                            PrintTableElement("Build");
                            PrintTableElement(output_string);
                            PrintEndTableRow();
                            if (ReadRegKeyString(HKEY_LOCAL_MACHINE,
                                                 "Software\\Microsoft\\Windo"
                                                 "ws NT\\CurrentVersion",
                                                 "BuildBranch", MAX_STRING_LENGTH - 1, output_string)) {
                                PrintBeginTableRow();
                                PrintTableElement("");
                                PrintTableElement("Branch");
                                PrintTableElement(output_string);
                                PrintEndTableRow();
                            }
                        }
                    } else {
                        PrintBeginTableRow();
                        PrintTableElement("Windows");
                        PrintTableElement("Windows 10 (or newer)");
                        PrintTableElement(os_size);
                        PrintEndTableRow();
                    }
                } else {
                    PrintBeginTableRow();
                    PrintTableElement("Windows");
                    PrintTableElement("Windows Server 2016 (or newer)");
                    PrintTableElement(os_size);
                    PrintEndTableRow();
                }
                break;
            case 6:
                switch (os_info.dwMinorVersion) {
                    case 3:
                        if (os_info.wProductType == VER_NT_WORKSTATION) {
                            if (ReadRegKeyString(HKEY_LOCAL_MACHINE, "Software\\Microsoft\\Windows NT\\CurrentVersion",
                                                 "ProductName", MAX_STRING_LENGTH - 1, generic_string)) {
                                PrintBeginTableRow();
                                PrintTableElement("Windows");
                                PrintTableElement(generic_string);
                                PrintTableElement(os_size);
                                PrintEndTableRow();

                                if (ReadRegKeyString(HKEY_LOCAL_MACHINE,
                                                     "Software\\Microsoft\\Windo"
                                                     "ws NT\\CurrentVersion",
                                                     "CurrentBuild", MAX_STRING_LENGTH - 1, output_string)) {
                                    PrintBeginTableRow();
                                    PrintTableElement("");
                                    PrintTableElement("Build");
                                    PrintTableElement(output_string);
                                    PrintEndTableRow();

                                    if (ReadRegKeyString(HKEY_LOCAL_MACHINE,
                                                         "Software\\Microsoft\\Windo"
                                                         "ws NT\\CurrentVersion",
                                                         "BuildBranch", MAX_STRING_LENGTH - 1, output_string)) {
                                        PrintBeginTableRow();
                                        PrintTableElement("");
                                        PrintTableElement("Branch");
                                        PrintTableElement(output_string);
                                        PrintEndTableRow();
                                    }
                                }
                            }
                        } else {
                            PrintBeginTableRow();
                            PrintTableElement("Windows");
                            PrintTableElement("Windows Server 2012 R2 (or newer)");
                            PrintTableElement(os_size);
                            PrintEndTableRow();
                        }
                        break;
                    case 2:
                        if (os_info.wProductType == VER_NT_WORKSTATION) {
                            if (ReadRegKeyString(HKEY_LOCAL_MACHINE, "Software\\Microsoft\\Windows NT\\CurrentVersion",
                                                 "ProductName", MAX_STRING_LENGTH - 1, generic_string)) {
                                PrintBeginTableRow();
                                PrintTableElement("Windows");
                                PrintTableElement(generic_string);
                                PrintTableElement(os_size);
                                PrintEndTableRow();

                                if (ReadRegKeyString(HKEY_LOCAL_MACHINE,
                                                     "Software\\Microsoft\\Windo"
                                                     "ws NT\\CurrentVersion",
                                                     "CurrentBuild", MAX_STRING_LENGTH - 1, output_string)) {
                                    PrintBeginTableRow();
                                    PrintTableElement("");
                                    PrintTableElement("Build");
                                    PrintTableElement(output_string);
                                    PrintEndTableRow();
                                    if (ReadRegKeyString(HKEY_LOCAL_MACHINE,
                                                         "Software\\Microsoft\\Windo"
                                                         "ws NT\\CurrentVersion",
                                                         "BuildBranch", MAX_STRING_LENGTH - 1, output_string)) {
                                        PrintBeginTableRow();
                                        PrintTableElement("");
                                        PrintTableElement("Branch");
                                        PrintTableElement(output_string);
                                        PrintEndTableRow();
                                    }
                                }
                            }
                        } else {
                            PrintBeginTableRow();
                            PrintTableElement("Windows");
                            PrintTableElement("Windows Server 2012 (or newer)");
                            PrintTableElement(os_size);
                            PrintEndTableRow();
                        }
                        break;
                    case 1:
                        if (os_info.wProductType == VER_NT_WORKSTATION) {
                            PrintBeginTableRow();
                            PrintTableElement("Windows");
                            PrintTableElement("Windows 7 (or newer)");
                            PrintTableElement(os_size);
                            PrintEndTableRow();
                        } else {
                            PrintBeginTableRow();
                            PrintTableElement("Windows");
                            PrintTableElement("Windows Server 2008 R2 (or newer)");
                            PrintTableElement(os_size);
                            PrintEndTableRow();
                        }
                        break;
                    default:
                        if (os_info.wProductType == VER_NT_WORKSTATION) {
                            PrintBeginTableRow();
                            PrintTableElement("Windows");
                            PrintTableElement("Windows Vista (or newer)");
                            PrintTableElement(os_size);
                            PrintEndTableRow();
                        } else {
                            PrintBeginTableRow();
                            PrintTableElement("Windows");
                            PrintTableElement("Windows Server 2008 (or newer)");
                            PrintTableElement(os_size);
                            PrintEndTableRow();
                        }
                        break;
                }
                break;
            case 5:
                ser_ver = GetSystemMetrics(SM_SERVERR2);
                switch (os_info.dwMinorVersion) {
                    case 2:
                        if ((os_info.wProductType == VER_NT_WORKSTATION) &&
                            (sys_info.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64)) {
                            strncpy(generic_string, "Windows XP Professional x64", MAX_STRING_LENGTH - 1);
                        } else if (os_info.wSuiteMask & VER_SUITE_WH_SERVER) {
                            strncpy(generic_string, "Windows Home Server", MAX_STRING_LENGTH - 1);
                        } else if (ser_ver != 0) {
                            strncpy(generic_string, "Windows Server 2003 R2", MAX_STRING_LENGTH - 1);
                        } else {
                            strncpy(generic_string, "Windows Server 2003", MAX_STRING_LENGTH - 1);
                        }
                        PrintBeginTableRow();
                        PrintTableElement("Windows");
                        PrintTableElement(generic_string);
                        PrintTableElement(os_size);
                        PrintEndTableRow();
                        break;
                    case 1:
                        PrintBeginTableRow();
                        PrintTableElement("Windows");
                        PrintTableElement("Windows XP");
                        PrintTableElement(os_size);
                        PrintEndTableRow();
                        break;
                    case 0:
                        PrintBeginTableRow();
                        PrintTableElement("Windows");
                        PrintTableElement("Windows 2000");
                        PrintTableElement(os_size);
                        PrintEndTableRow();
                        break;
                    default:
                        PrintBeginTableRow();
                        PrintTableElement("Windows");
                        PrintTableElement("Unknown Windows OS");
                        PrintTableElement(os_size);
                        PrintEndTableRow();
                        break;
                }
                break;
        }
    } else {
        PrintBeginTableRow();
        PrintTableElement("Windows");
        PrintTableElement("Error retrieving Windows Version");
        PrintTableElement("");
        PrintEndTableRow();
        res = UNKNOWN_ERROR;
        goto out;
    }

    if (0 != GetEnvironmentVariableA("SYSTEMROOT", system_root_dir, MAX_STRING_LENGTH - 1)) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("System Root");
        PrintTableElement(system_root_dir);
        PrintEndTableRow();
    }
    if (0 != GetEnvironmentVariableA("PROGRAMDATA", generic_string, MAX_STRING_LENGTH - 1)) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("Program Data");
        PrintTableElement(generic_string);
        PrintEndTableRow();
    }
    if (0 != GetEnvironmentVariableA("PROGRAMFILES", generic_string, MAX_STRING_LENGTH - 1)) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("Program Files");
        PrintTableElement(generic_string);
        PrintEndTableRow();
    }
    if (0 != GetEnvironmentVariableA("PROGRAMFILES(X86)", generic_string, MAX_STRING_LENGTH - 1)) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("Program Files (x86)");
        PrintTableElement(generic_string);
        PrintEndTableRow();
    }
    if (0 != GetEnvironmentVariableA("TEMP", generic_string, MAX_STRING_LENGTH - 1)) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("TEMP");
        PrintTableElement(generic_string);
        PrintEndTableRow();
    }
    if (0 != GetEnvironmentVariableA("TMP", generic_string, MAX_STRING_LENGTH - 1)) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("TMP");
        PrintTableElement(generic_string);
        PrintEndTableRow();
    }

    PrintEndTable();

    // Output whatever generic hardware information we can find out about the
    // system.  Including how much memory and disk space is available.
    PrintBeginTable("Hardware", 3);

    snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u", sys_info.dwNumberOfProcessors);
    PrintBeginTableRow();
    PrintTableElement("CPUs");
    PrintTableElement("Number of Logical Cores");
    PrintTableElement(generic_string);
    PrintEndTableRow();

    switch (sys_info.wProcessorArchitecture) {
        case PROCESSOR_ARCHITECTURE_AMD64:
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Type");
            PrintTableElement("x86_64");
            PrintEndTableRow();
            break;
        case PROCESSOR_ARCHITECTURE_ARM:
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Type");
            PrintTableElement("ARM");
            PrintEndTableRow();
            break;
        case PROCESSOR_ARCHITECTURE_IA64:
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Type");
            PrintTableElement("IA64");
            PrintEndTableRow();
            break;
        case PROCESSOR_ARCHITECTURE_INTEL:
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Type");
            PrintTableElement("x86");
            PrintEndTableRow();
            break;
        default:
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Type");
            PrintTableElement("Unknown");
            PrintEndTableRow();
            break;
    }

    if (TRUE == GlobalMemoryStatusEx(&mem_stat)) {
        if ((mem_stat.ullTotalPhys >> 40) > 0x0ULL) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u TB", static_cast<uint32_t>(mem_stat.ullTotalPhys >> 40));
            PrintBeginTableRow();
            PrintTableElement("Memory");
            PrintTableElement("Physical");
            PrintTableElement(generic_string);
            PrintEndTableRow();
        } else if ((mem_stat.ullTotalPhys >> 30) > 0x0ULL) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u GB", static_cast<uint32_t>(mem_stat.ullTotalPhys >> 30));
            PrintBeginTableRow();
            PrintTableElement("Memory");
            PrintTableElement("Physical");
            PrintTableElement(generic_string);
            PrintEndTableRow();
        } else if ((mem_stat.ullTotalPhys >> 20) > 0x0ULL) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u MB", static_cast<uint32_t>(mem_stat.ullTotalPhys >> 20));
            PrintBeginTableRow();
            PrintTableElement("Memory");
            PrintTableElement("Physical");
            PrintTableElement(generic_string);
            PrintEndTableRow();
        } else if ((mem_stat.ullTotalPhys >> 10) > 0x0ULL) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u KB", static_cast<uint32_t>(mem_stat.ullTotalPhys >> 10));
            PrintBeginTableRow();
            PrintTableElement("Memory");
            PrintTableElement("Physical");
            PrintTableElement(generic_string);
            PrintEndTableRow();
        } else {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u bytes", static_cast<uint32_t>(mem_stat.ullTotalPhys));
            PrintBeginTableRow();
            PrintTableElement("Memory");
            PrintTableElement("Physical");
            PrintTableElement(generic_string);
            PrintEndTableRow();
        }
    }

    if (TRUE == GetDiskFreeSpaceA(NULL, &sect_per_cluster, &bytes_per_sect, &num_free_cluster, &total_num_cluster)) {
        uint64_t bytes_free = (uint64_t)bytes_per_sect * (uint64_t)sect_per_cluster * (uint64_t)num_free_cluster;
        uint64_t bytes_total = (uint64_t)bytes_per_sect * (uint64_t)sect_per_cluster * (uint64_t)total_num_cluster;
        double perc_free = (double)bytes_free / (double)bytes_total;
        if ((bytes_total >> 40) > 0x0ULL) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u TB", static_cast<uint32_t>(bytes_total >> 40));
            PrintBeginTableRow();
            PrintTableElement("Disk Space");
            PrintTableElement("Total");
            PrintTableElement(generic_string);
            PrintEndTableRow();
        } else if ((bytes_total >> 30) > 0x0ULL) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u GB", static_cast<uint32_t>(bytes_total >> 30));
            PrintBeginTableRow();
            PrintTableElement("Disk Space");
            PrintTableElement("Total");
            PrintTableElement(generic_string);
            PrintEndTableRow();
        } else if ((bytes_total >> 20) > 0x0ULL) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u MB", static_cast<uint32_t>(bytes_total >> 20));
            PrintBeginTableRow();
            PrintTableElement("Disk Space");
            PrintTableElement("Total");
            PrintTableElement(generic_string);
            PrintEndTableRow();
        } else if ((bytes_total >> 10) > 0x0ULL) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u KB", static_cast<uint32_t>(bytes_total >> 10));
            PrintBeginTableRow();
            PrintTableElement("Disk Space");
            PrintTableElement("Total");
            PrintTableElement(generic_string);
            PrintEndTableRow();
        }
        snprintf(output_string, MAX_STRING_LENGTH - 1, "%4.2f%%", (static_cast<float>(perc_free) * 100.f));
        if ((bytes_free >> 40) > 0x0ULL) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u TB", static_cast<uint32_t>(bytes_free >> 40));
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Free");
            PrintTableElement(generic_string);
            PrintEndTableRow();
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Free Perc");
            PrintTableElement(output_string);
            PrintEndTableRow();
        } else if ((bytes_free >> 30) > 0x0ULL) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u GB", static_cast<uint32_t>(bytes_free >> 30));
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Free");
            PrintTableElement(generic_string);
            PrintEndTableRow();
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Free Perc");
            PrintTableElement(output_string);
            PrintEndTableRow();
        } else if ((bytes_free >> 20) > 0x0ULL) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u MB", static_cast<uint32_t>(bytes_free >> 20));
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Free");
            PrintTableElement(generic_string);
            PrintEndTableRow();
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Free Perc");
            PrintTableElement(output_string);
            PrintEndTableRow();
        } else if ((bytes_free >> 10) > 0x0ULL) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u KB", static_cast<uint32_t>(bytes_free >> 10));
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Free");
            PrintTableElement(generic_string);
            PrintEndTableRow();
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Free Perc");
            PrintTableElement(output_string);
            PrintEndTableRow();
        }
    }

    PrintEndTable();

    // Print out information about this executable.
    PrintBeginTable("Executable", 2);

    PrintBeginTableRow();
    PrintTableElement("Exe Directory");
    PrintTableElement(global_items.exe_directory);
    PrintEndTableRow();

    if (0 != GetCurrentDirectoryA(MAX_STRING_LENGTH - 1, generic_string)) {
        cur_directory = generic_string;
        PrintBeginTableRow();
        PrintTableElement("Current Directory");
        PrintTableElement(generic_string);
        PrintEndTableRow();
    } else {
        cur_directory = "";
    }

    PrintBeginTableRow();
    PrintTableElement("Vulkan API Version");
    uint32_t major = VK_VERSION_MAJOR(VK_API_VERSION_1_0);
    uint32_t minor = VK_VERSION_MINOR(VK_API_VERSION_1_0);
    uint32_t patch = VK_VERSION_PATCH(VK_HEADER_VERSION);
    snprintf(generic_string, MAX_STRING_LENGTH - 1, "%d.%d.%d", major, minor, patch);
    PrintTableElement(generic_string);
    PrintEndTableRow();

    PrintBeginTableRow();
    PrintTableElement("Byte Format");
#if _WIN64 || __x86_64__ || __ppc64__
    PrintTableElement("64-bit");
#else
    PrintTableElement("32-bit");
#endif
    PrintEndTableRow();

    PrintEndTable();

    // Now print out the remaining system info.
    res = PrintDriverInfo();
    if (res != SUCCESSFUL) {
        goto out;
    }
    PrintRunTimeInfo();
    res = PrintSDKInfo();
    res = PrintLayerInfo();
    res = PrintLayerSettingsFileInfo();
    EndSection();

out:

    return res;
}

// Determine what version an executable or library file is.
bool GetFileVersion(const char *filename, const uint32_t max_len, char *version_string) {
    DWORD ver_handle;
    UINT size = 0;
    LPBYTE buffer = NULL;
    DWORD ver_size = GetFileVersionInfoSize(filename, &ver_handle);
    bool success = false;

    if (ver_size > 0) {
        LPSTR ver_data = (LPSTR)malloc(sizeof(char) * ver_size);

        if (GetFileVersionInfo(filename, ver_handle, ver_size, ver_data)) {
            if (VerQueryValue(ver_data, "\\", (VOID FAR * FAR *)&buffer, &size)) {
                if (size) {
                    VS_FIXEDFILEINFO *ver_info = (VS_FIXEDFILEINFO *)buffer;
                    if (ver_info->dwSignature == 0xfeef04bd) {
                        DWORD max_size = ver_size > max_len ? max_len : ver_size;
                        snprintf(version_string, max_len, "%d.%d.%d.%d", (ver_info->dwFileVersionMS >> 16) & 0xffff,
                                 (ver_info->dwFileVersionMS >> 0) & 0xffff, (ver_info->dwFileVersionLS >> 16) & 0xffff,
                                 (ver_info->dwFileVersionLS >> 0) & 0xffff);
                        success = true;
                    }
                }
            }
        }
        free(ver_data);
    }

    return success;
}

bool ReadDriverJson(std::string cur_driver_json, std::string system_path, bool &found_lib) {
    bool found_json = false;
    std::ifstream *stream = NULL;
    Json::Value root = Json::nullValue;
    Json::Value dev_exts = Json::nullValue;
    Json::Value inst_exts = Json::nullValue;
    Json::Reader reader;
    char full_driver_path[MAX_STRING_LENGTH];
    char generic_string[MAX_STRING_LENGTH];
    uint32_t j = 0;

    stream = new std::ifstream(cur_driver_json.c_str(), std::ifstream::in);
    if (nullptr == stream || stream->fail()) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("Error reading JSON file");
        PrintTableElement(cur_driver_json);
        PrintEndTableRow();
        goto out;
    }

    if (!reader.parse(*stream, root, false) || root.isNull()) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("Error reading JSON file");
        PrintTableElement(reader.getFormattedErrorMessages());
        PrintEndTableRow();
        goto out;
    }

    PrintBeginTableRow();
    PrintTableElement("");
    PrintTableElement("JSON File Version");
    if (!root["file_format_version"].isNull()) {
        PrintTableElement(root["file_format_version"].asString());
    } else {
        PrintTableElement("MISSING!");
    }
    PrintEndTableRow();

    if (root["ICD"].isNull()) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("ICD Section");
        PrintTableElement("MISSING!");
        PrintEndTableRow();
        goto out;
    }

    found_json = true;

    PrintBeginTableRow();
    PrintTableElement("");
    PrintTableElement("API Version");
    if (!root["ICD"]["api_version"].isNull()) {
        PrintTableElement(root["ICD"]["api_version"].asString());
    } else {
        PrintTableElement("MISSING!");
    }
    PrintEndTableRow();

    PrintBeginTableRow();
    PrintTableElement("");
    PrintTableElement("Library Path");
    if (!root["ICD"]["library_path"].isNull()) {
        std::string driver_name = root["ICD"]["library_path"].asString();
        PrintTableElement(driver_name);
        PrintEndTableRow();

        if (GenerateLibraryPath(cur_driver_json.c_str(), driver_name.c_str(), MAX_STRING_LENGTH, full_driver_path)) {
            std::string system_name = system_path;
            system_name += "\\";
            system_name += driver_name;

            if (GetFileVersion(full_driver_path, MAX_STRING_LENGTH - 1, generic_string)) {
                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement("Library File Version");
                PrintTableElement(generic_string);
                PrintEndTableRow();

                found_lib = true;
            } else if (GetFileVersion(system_name.c_str(), MAX_STRING_LENGTH - 1, generic_string)) {
                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement("Library File Version");
                PrintTableElement(generic_string);
                PrintEndTableRow();

                found_lib = true;
            } else {
                snprintf(generic_string, MAX_STRING_LENGTH - 1,
                         "Failed to find driver %s "
                         " or %sreferenced by JSON %s",
                         root["ICD"]["library_path"].asString().c_str(), full_driver_path, cur_driver_json.c_str());
                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement("");
                PrintTableElement(generic_string);
                PrintEndTableRow();
            }
        } else {
            snprintf(generic_string, MAX_STRING_LENGTH - 1,
                     "Failed to find driver %s "
                     "referenced by JSON %s",
                     full_driver_path, cur_driver_json.c_str());
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("");
            PrintTableElement(generic_string);
            PrintEndTableRow();
        }
    } else {
        PrintTableElement("MISSING!");
        PrintEndTableRow();
    }

    char count_str[MAX_STRING_LENGTH];
    j = 0;
    dev_exts = root["ICD"]["device_extensions"];
    if (!dev_exts.isNull() && dev_exts.isArray()) {
        snprintf(count_str, MAX_STRING_LENGTH - 1, "%d", dev_exts.size());
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("Device Extensions");
        PrintTableElement(count_str);
        PrintEndTableRow();

        for (Json::ValueIterator dev_ext_it = dev_exts.begin(); dev_ext_it != dev_exts.end(); dev_ext_it++) {
            Json::Value dev_ext = (*dev_ext_it);
            Json::Value dev_ext_name = dev_ext["name"];
            if (!dev_ext_name.isNull()) {
                snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", j);

                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement(generic_string, ALIGN_RIGHT);
                PrintTableElement(dev_ext_name.asString());
                PrintEndTableRow();
            }
        }
    }
    inst_exts = root["ICD"]["instance_extensions"];
    j = 0;
    if (!inst_exts.isNull() && inst_exts.isArray()) {
        snprintf(count_str, MAX_STRING_LENGTH - 1, "%d", inst_exts.size());
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("Instance Extensions");
        PrintTableElement(count_str);
        PrintEndTableRow();

        for (Json::ValueIterator inst_ext_it =

                 inst_exts.begin();
             inst_ext_it != inst_exts.end(); inst_ext_it++) {
            Json::Value inst_ext = (*inst_ext_it);
            Json::Value inst_ext_name = inst_ext["name"];
            if (!inst_ext_name.isNull()) {
                snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", j);

                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement(generic_string, ALIGN_RIGHT);
                PrintTableElement(inst_ext_name.asString());
                PrintEndTableRow();
            }
        }
    }

out:

    if (nullptr != stream) {
        stream->close();
        delete stream;
        stream = NULL;
    }

    return found_json;
}

void PrintDriverRegInfo(HKEY reg_folder, const char *reg_key_loc, const char *system_path, bool found_this_lib,
                        char *cur_vulkan_driver_json, char *generic_string, bool &found_registry, bool &found_json,
                        bool &found_lib) {
    // Find the registry settings indicating the location of the driver
    // JSON files.
    uint32_t i = 0;
    uint32_t returned_value = 0;
    while (FindNextRegValue(reg_folder, reg_key_loc, "", i, MAX_STRING_LENGTH - 1, cur_vulkan_driver_json, &returned_value)) {
        found_registry |= true;

        snprintf(generic_string, MAX_STRING_LENGTH - 1, "Driver %d", i++);

        PrintBeginTableRow();
        PrintTableElement(generic_string, ALIGN_RIGHT);
        PrintTableElement(cur_vulkan_driver_json);

        if (returned_value != 0) {
            PrintTableElement("DISABLED");
        } else {
            PrintTableElement("ENABLED");
        }
        PrintEndTableRow();

        // Parse the driver JSON file.
        if (ReadDriverJson(cur_vulkan_driver_json, system_path, found_this_lib)) {
            found_json |= true;
            found_lib |= found_this_lib;
        }
    }
}

// Print out the information for every driver in the appropriate
// Windows registry location and its corresponding JSON file.
ErrorResults PrintDriverInfo(void) {
    ErrorResults res = SUCCESSFUL;
    const char vulkan_reg_base[] = "SOFTWARE\\Khronos\\Vulkan";
    const char vulkan_reg_base_wow64[] = "SOFTWARE\\WOW6432Node\\Khronos\\Vulkan";
    char reg_key_loc[MAX_STRING_LENGTH];
    char cur_vulkan_driver_json[MAX_STRING_LENGTH];
    char generic_string[MAX_STRING_LENGTH];
    char system_path[MAX_STRING_LENGTH];
    char env_value[MAX_STRING_LENGTH];
    uint32_t i = 0;
    std::ifstream *stream = NULL;
    bool found_registry = false;
    bool found_json = false;
    bool found_lib = false;
    bool found_this_lib = false;

    GetEnvironmentVariableA("SYSTEMROOT", generic_string, MAX_STRING_LENGTH);
#if _WIN64 || __x86_64__ || __ppc64__
    snprintf(system_path, MAX_STRING_LENGTH - 1, "%s\\system32\\", generic_string);
    snprintf(reg_key_loc, MAX_STRING_LENGTH - 1, "%s\\Drivers", vulkan_reg_base);
#else
    if (global_items.is_wow64) {
        snprintf(system_path, MAX_STRING_LENGTH - 1, "%s\\sysWOW64\\", generic_string);
        snprintf(reg_key_loc, MAX_STRING_LENGTH - 1, "%s\\Drivers", vulkan_reg_base_wow64);
    } else {
        snprintf(system_path, MAX_STRING_LENGTH - 1, "%s\\system32\\", generic_string);
        snprintf(reg_key_loc, MAX_STRING_LENGTH - 1, "%s\\Drivers", vulkan_reg_base);
    }
#endif

    PrintBeginTable("Vulkan Driver Info", 3);
    PrintBeginTableRow();
    PrintTableElement("Drivers in Registry");
    PrintTableElement(reg_key_loc);
    PrintTableElement("");
    PrintEndTableRow();

    PrintDriverRegInfo(HKEY_LOCAL_MACHINE, reg_key_loc, system_path, found_this_lib, cur_vulkan_driver_json, generic_string,
                       found_registry, found_json, found_lib);
    PrintDriverRegInfo(HKEY_CURRENT_USER, reg_key_loc, system_path, found_this_lib, cur_vulkan_driver_json, generic_string,
                       found_registry, found_json, found_lib);

    // The user can override the drivers path manually
    if (0 != GetEnvironmentVariableA("VK_DRIVERS_PATH", env_value, MAX_STRING_LENGTH - 1) && 0 != strlen(env_value)) {
        WIN32_FIND_DATAA ffd;
        HANDLE hFind;
        char *tok = NULL;
        bool keep_looping = false;
        char full_driver_path[MAX_STRING_LENGTH];
        char cur_driver_path[MAX_STRING_LENGTH];
        uint32_t path = 0;

        PrintBeginTableRow();
        PrintTableElement("VK_DRIVERS_PATH");
        PrintTableElement(env_value);
        PrintTableElement("");
        PrintEndTableRow();

        tok = strtok(env_value, ";");
        if (NULL != tok) {
            keep_looping = true;
            strncpy(cur_driver_path, tok, MAX_STRING_LENGTH - 1);
        } else {
            strncpy(cur_driver_path, env_value, MAX_STRING_LENGTH - 1);
        }

        do {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "Path %d", path++);
            PrintBeginTableRow();
            PrintTableElement(generic_string, ALIGN_CENTER);
            PrintTableElement(cur_driver_path);
            PrintTableElement("");
            PrintEndTableRow();

            // Look for any JSON files in that folder.
            snprintf(full_driver_path, MAX_STRING_LENGTH - 1, "%s\\*.json", cur_driver_path);
            hFind = FindFirstFileA(full_driver_path, &ffd);
            if (hFind != INVALID_HANDLE_VALUE) {
                do {
                    if (0 == (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                        snprintf(generic_string, MAX_STRING_LENGTH - 1, "Driver %d", i++);
                        snprintf(cur_vulkan_driver_json, MAX_STRING_LENGTH - 1, "%s\\%s", cur_driver_path, ffd.cFileName);

                        PrintBeginTableRow();
                        PrintTableElement(generic_string, ALIGN_RIGHT);
                        PrintTableElement(ffd.cFileName);
                        PrintTableElement("");
                        PrintEndTableRow();

                        // Parse the driver JSON file.
                        if (ReadDriverJson(cur_vulkan_driver_json, system_path, found_this_lib)) {
                            found_json = true;
                            found_lib |= found_this_lib;
                        }
                    }
                } while (FindNextFileA(hFind, &ffd) != 0);
                FindClose(hFind);
            }

            tok = strtok(NULL, ";");
            if (NULL == tok) {
                keep_looping = false;
            } else {
                strncpy(cur_driver_path, tok, MAX_STRING_LENGTH - 1);
            }
        } while (keep_looping);
    }

    // The user can override the driver file manually
    if (0 != GetEnvironmentVariableA("VK_ICD_FILENAMES", env_value, MAX_STRING_LENGTH - 1) && 0 != strlen(env_value)) {
        WIN32_FIND_DATAA ffd;
        HANDLE hFind;
        char *tok = NULL;
        bool keep_looping = false;
        char full_driver_path[MAX_STRING_LENGTH];

        PrintBeginTableRow();
        PrintTableElement("VK_ICD_FILENAMES");
        PrintTableElement(env_value);
        PrintTableElement("");
        PrintEndTableRow();

        tok = strtok(env_value, ";");
        if (NULL != tok) {
            keep_looping = true;
            strncpy(full_driver_path, tok, MAX_STRING_LENGTH - 1);
        } else {
            strncpy(full_driver_path, env_value, MAX_STRING_LENGTH - 1);
        }

        do {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "Driver %d", i++);
            PrintBeginTableRow();
            PrintTableElement(generic_string, ALIGN_RIGHT);
            PrintTableElement(full_driver_path);
            PrintTableElement("");
            PrintEndTableRow();

            hFind = FindFirstFileA(full_driver_path, &ffd);
            if (hFind != INVALID_HANDLE_VALUE) {
                if (0 == (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    strcpy(cur_vulkan_driver_json, full_driver_path);
                    // Parse the driver JSON file.
                    if (ReadDriverJson(cur_vulkan_driver_json, system_path, found_this_lib)) {
                        found_json = true;
                        found_lib |= found_this_lib;
                    }
                }
                FindClose(hFind);
            } else {
                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement("Driver Not Found");
                PrintTableElement("");
                PrintEndTableRow();
            }

            tok = strtok(NULL, ";");
            if (NULL == tok) {
                keep_looping = false;
            } else {
                strncpy(full_driver_path, tok, MAX_STRING_LENGTH - 1);
            }
        } while (keep_looping);
    }

    PrintEndTable();

    if (!found_registry) {
        res = MISSING_DRIVER_REGISTRY;
    } else if (!found_json) {
        res = MISSING_DRIVER_JSON;
    } else if (!found_lib) {
        res = MISSING_DRIVER_LIB;
    }

    return res;
}

void PrintUninstallRegInfo(HKEY reg_folder, char *output_string, char *count_string, char *generic_string, char *version_string,
                           unsigned int &install_count) {
    uint32_t i = 0;
    // Find all Vulkan Runtime keys in the registry, and loop through each.
    while (FindNextRegKey(reg_folder, g_uninstall_reg_path, "VulkanRT", i, MAX_STRING_LENGTH - 1, output_string)) {
        snprintf(count_string, MAX_STRING_LENGTH - 1, "[%d]", i++);

        snprintf(generic_string, MAX_STRING_LENGTH - 1, "%s\\%s", g_uninstall_reg_path, output_string);

        // Get the version from the registry
        if (!ReadRegKeyString(reg_folder, generic_string, "DisplayVersion", MAX_STRING_LENGTH - 1, version_string)) {
            strncpy(version_string, output_string, MAX_STRING_LENGTH - 1);
        }

        // Get the install count for this runtime from the registry
        if (ReadRegKeyDword(reg_folder, generic_string, "InstallCount", &install_count)) {
            snprintf(output_string, MAX_STRING_LENGTH - 1, "%s  [Install Count = %d]", version_string, install_count);
        } else {
            snprintf(output_string, MAX_STRING_LENGTH - 1, "%s", version_string);
        }
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement(count_string, ALIGN_RIGHT);
        PrintTableElement(output_string);
        PrintEndTableRow();
    }
}

// Print out whatever Vulkan runtime information we can gather from the system
// using the registry, standard system paths, etc.
ErrorResults PrintRunTimeInfo(void) {
    ErrorResults res = SUCCESSFUL;
    char generic_string[MAX_STRING_LENGTH];
    char count_string[MAX_STRING_LENGTH];
    char version_string[MAX_STRING_LENGTH];
    char output_string[MAX_STRING_LENGTH];
    char dll_search[MAX_STRING_LENGTH];
    char dll_prefix[MAX_STRING_LENGTH];
    uint32_t i = 0;
    uint32_t install_count = 0;
    FILE *fp = NULL;
    bool found = false;

    PrintBeginTable("Vulkan Runtimes", 3);

    PrintBeginTableRow();
    PrintTableElement("Runtimes In Registry");
    PrintTableElement(g_uninstall_reg_path);
    PrintTableElement("");
    PrintEndTableRow();

    PrintUninstallRegInfo(HKEY_LOCAL_MACHINE, output_string, count_string, generic_string, version_string, install_count);
    PrintUninstallRegInfo(HKEY_CURRENT_USER, output_string, count_string, generic_string, version_string, install_count);

    i = 0;
    GetEnvironmentVariableA("SYSTEMROOT", generic_string, MAX_STRING_LENGTH);
#if _WIN64 || __x86_64__ || __ppc64__
    snprintf(dll_prefix, MAX_STRING_LENGTH - 1, "%s\\system32\\", generic_string);
#else
    if (global_items.is_wow64) {
        snprintf(dll_prefix, MAX_STRING_LENGTH - 1, "%s\\sysWOW64\\", generic_string);
    } else {
        snprintf(dll_prefix, MAX_STRING_LENGTH - 1, "%s\\system32\\", generic_string);
    }
#endif

    PrintBeginTableRow();
    PrintTableElement("Runtimes in System Folder");
    PrintTableElement(dll_prefix);
    PrintTableElement("");
    PrintEndTableRow();

    strncpy(dll_search, dll_prefix, MAX_STRING_LENGTH - 1);
    strncat(dll_search, "Vulkan-*.dll", MAX_STRING_LENGTH - 1);

    WIN32_FIND_DATAA ffd;
    HANDLE hFind = FindFirstFileA(dll_search, &ffd);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (0 == (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                snprintf(count_string, MAX_STRING_LENGTH - 1, "DLL %d", i++);

                PrintBeginTableRow();
                PrintTableElement(count_string, ALIGN_RIGHT);
                PrintTableElement(ffd.cFileName);

                snprintf(generic_string, MAX_STRING_LENGTH - 1, "%s\\%s", dll_prefix, ffd.cFileName);
                if (GetFileVersion(generic_string, MAX_STRING_LENGTH - 1, version_string)) {
                    snprintf(output_string, MAX_STRING_LENGTH - 1, "Version %s", version_string);
                    PrintTableElement(output_string);
                } else {
                    PrintTableElement("");
                }
                PrintEndTableRow();
            }
        } while (FindNextFileA(hFind, &ffd) != 0);
        FindClose(hFind);
    }

    PrintBeginTableRow();
    PrintTableElement("Runtime Used by App");
    if (!system("where vulkan-1.dll > where_vulkan")) {
        fp = fopen("where_vulkan", "rt");
        if (NULL != fp) {
            if (NULL != fgets(generic_string, MAX_STRING_LENGTH - 1, fp)) {
                int cur_char = (int)strlen(generic_string) - 1;
                while (generic_string[cur_char] == '\n' || generic_string[cur_char] == '\r' || generic_string[cur_char] == '\t' ||
                       generic_string[cur_char] == ' ') {
                    generic_string[cur_char] = '\0';
                    cur_char--;
                }

                if (GetFileVersion(generic_string, MAX_STRING_LENGTH - 1, version_string)) {
                    PrintTableElement(generic_string);
                    PrintTableElement(version_string);
                } else {
                    PrintTableElement(generic_string);
                    PrintTableElement("");
                }
                found = true;
            }
            fclose(fp);
        }
        DeleteFileA("where_vulkan");
    } else {
        PrintTableElement("Unknown");
        PrintTableElement("Unknown");
    }
    PrintEndTableRow();

    PrintEndTable();

    if (!found) {
        res = VULKAN_CANT_FIND_RUNTIME;
    }

    return res;
}

bool PrintSdkUninstallRegInfo(HKEY reg_folder, char *output_string, char *count_string, char *generic_string) {
    uint32_t i = 0;
    bool found = false;
    while (FindNextRegKey(reg_folder, g_uninstall_reg_path, "VulkanSDK", i, MAX_STRING_LENGTH, output_string)) {
        found = true;
        snprintf(count_string, MAX_STRING_LENGTH - 1, "[%d]", i++);
        snprintf(generic_string, MAX_STRING_LENGTH - 1, "%s\\%s", g_uninstall_reg_path, output_string);
        ReadRegKeyString(reg_folder, generic_string, "InstallDir", MAX_STRING_LENGTH, output_string);

        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement(count_string, ALIGN_RIGHT);
        PrintTableElement(output_string);
        PrintEndTableRow();
    }
    return found;
}

bool PrintExplicitLayersRegInfo(HKEY reg_folder, const char *reg_key_loc, const char *sdk_env_dir, char *output_string,
                                char *count_string, char *cur_vulkan_layer_json, ErrorResults &res) {
    bool found = false;
    uint32_t i = 0;
    uint32_t returned_value = 0;
    while (FindNextRegValue(reg_folder, reg_key_loc, "", i, MAX_STRING_LENGTH, cur_vulkan_layer_json, &returned_value)) {
        found = true;

        // Create a short json file name so we don't use up too much space
        snprintf(output_string, MAX_STRING_LENGTH - 1, ".%s", &cur_vulkan_layer_json[strlen(sdk_env_dir)]);

        snprintf(count_string, MAX_STRING_LENGTH - 1, "[%d]", i++);
        PrintBeginTableRow();
        PrintTableElement(count_string, ALIGN_RIGHT);
        PrintTableElement(output_string);

        snprintf(output_string, MAX_STRING_LENGTH - 1, "0x%08x", returned_value);
        PrintTableElement(output_string);
        PrintEndTableRow();

        std::ifstream *stream = NULL;
        stream = new std::ifstream(cur_vulkan_layer_json, std::ifstream::in);
        if (nullptr == stream || stream->fail()) {
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("ERROR reading JSON file!");
            PrintTableElement("");
            PrintEndTableRow();
            res = MISSING_LAYER_JSON;
        } else {
            Json::Value root = Json::nullValue;
            Json::Reader reader;
            if (!reader.parse(*stream, root, false) || root.isNull()) {
                // Report to the user the failure and their locations in the
                // document.
                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement("ERROR parsing JSON file!");
                PrintTableElement(reader.getFormattedErrorMessages());
                PrintEndTableRow();
                res = LAYER_JSON_PARSING_ERROR;
            } else {
                PrintExplicitLayerJsonInfo(cur_vulkan_layer_json, root, 3);
            }

            stream->close();
            delete stream;
            stream = NULL;
        }
    }
    return found;
}

// Print out information on whatever LunarG Vulkan SDKs we can find on
// the system using the registry, and environmental variables.  This
// includes listing what layers are available from the SDK.
ErrorResults PrintSDKInfo(void) {
    ErrorResults res = SUCCESSFUL;
    const char vulkan_reg_base[] = "SOFTWARE\\Khronos\\Vulkan";
    const char vulkan_reg_base_wow64[] = "SOFTWARE\\WOW6432Node\\Khronos\\Vulkan";
    char generic_string[MAX_STRING_LENGTH];
    char count_string[MAX_STRING_LENGTH];
    char output_string[MAX_STRING_LENGTH];
    char cur_vulkan_layer_json[MAX_STRING_LENGTH];
    char sdk_env_dir[MAX_STRING_LENGTH];
    char reg_key_loc[MAX_STRING_LENGTH];
    uint32_t i = 0;
    uint32_t j = 0;
    FILE *fp = NULL;
    bool found = false;

    PrintBeginTable("LunarG Vulkan SDKs", 3);
    PrintBeginTableRow();
    PrintTableElement("SDKs Found In Registry");
    PrintTableElement(g_uninstall_reg_path);
    PrintTableElement("");
    PrintEndTableRow();

    found |= PrintSdkUninstallRegInfo(HKEY_LOCAL_MACHINE, output_string, count_string, generic_string);
    found |= PrintSdkUninstallRegInfo(HKEY_CURRENT_USER, output_string, count_string, generic_string);

    if (!found) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("NONE FOUND", ALIGN_RIGHT);
        PrintTableElement("");
        PrintEndTableRow();
    }

    if (0 != GetEnvironmentVariableA("VK_SDK_PATH", sdk_env_dir, MAX_STRING_LENGTH - 1)) {
        PrintBeginTableRow();
        PrintTableElement("VK_SDK_PATH");
        global_items.sdk_found = true;
        global_items.sdk_path = sdk_env_dir;
        PrintTableElement(sdk_env_dir);
        PrintTableElement("");
        PrintEndTableRow();
    } else if (0 != GetEnvironmentVariableA("VULKAN_SDK", sdk_env_dir, MAX_STRING_LENGTH - 1)) {
        PrintBeginTableRow();
        PrintTableElement("VULKAN_SDK");
        global_items.sdk_found = true;
        global_items.sdk_path = sdk_env_dir;
        PrintTableElement(sdk_env_dir);
        PrintTableElement("");
        PrintEndTableRow();
    } else {
        PrintBeginTableRow();
        PrintTableElement("VK_SDK_PATH");
        PrintTableElement("No installed SDK");
        PrintTableElement("");
        PrintEndTableRow();
    }

#if _WIN64 || __x86_64__ || __ppc64__
    snprintf(reg_key_loc, MAX_STRING_LENGTH - 1, "%s\\ExplicitLayers", vulkan_reg_base);
#else
    if (global_items.is_wow64) {
        snprintf(reg_key_loc, MAX_STRING_LENGTH - 1, "%s\\ExplicitLayers", vulkan_reg_base_wow64);
    } else {
        snprintf(reg_key_loc, MAX_STRING_LENGTH - 1, "%s\\ExplicitLayers", vulkan_reg_base);
    }
#endif

    PrintBeginTableRow();
    PrintTableElement("SDK Explicit Layers");
    PrintTableElement(generic_string);
    PrintTableElement("");
    PrintEndTableRow();

    found = false;
    found |= PrintExplicitLayersRegInfo(HKEY_LOCAL_MACHINE, reg_key_loc, sdk_env_dir, output_string, count_string,
                                        cur_vulkan_layer_json, res);
    found |= PrintExplicitLayersRegInfo(HKEY_CURRENT_USER, reg_key_loc, sdk_env_dir, output_string, count_string,
                                        cur_vulkan_layer_json, res);

    if (!found) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("NONE FOUND", ALIGN_RIGHT);
        PrintTableElement("");
        PrintEndTableRow();
    }

    PrintEndTable();

    return res;
}

void PrintImplicitLayersRegInfo(HKEY reg_folder, const char *vulkan_impl_layer_reg_key, char *cur_vulkan_layer_json,
                                char *generic_string, ErrorResults &res) {
    // For each implicit layer listed in the registry, find its JSON and
    // print out the useful information stored in it.
    uint32_t i = 0;
    uint32_t returned_value = 0;
    while (
        FindNextRegValue(reg_folder, vulkan_impl_layer_reg_key, "", i, MAX_STRING_LENGTH, cur_vulkan_layer_json, &returned_value)) {
        snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", i++);

        PrintBeginTableRow();
        PrintTableElement(generic_string, ALIGN_RIGHT);
        PrintTableElement(cur_vulkan_layer_json);
        PrintTableElement("");
        snprintf(generic_string, MAX_STRING_LENGTH - 1, "0x%08x", returned_value);
        PrintTableElement(generic_string);
        PrintEndTableRow();

        std::ifstream *stream = NULL;
        stream = new std::ifstream(cur_vulkan_layer_json, std::ifstream::in);
        if (nullptr == stream || stream->fail()) {
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("ERROR reading JSON file!");
            PrintTableElement("");
            PrintEndTableRow();
            res = MISSING_LAYER_JSON;
        } else {
            Json::Value root = Json::nullValue;
            Json::Reader reader;
            if (!reader.parse(*stream, root, false) || root.isNull()) {
                // Report to the user the failure and their locations in the
                // document.
                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement("ERROR parsing JSON file!");
                PrintTableElement(reader.getFormattedErrorMessages());
                PrintEndTableRow();
                res = LAYER_JSON_PARSING_ERROR;
            } else {
                PrintImplicitLayerJsonInfo(cur_vulkan_layer_json, root);
            }

            stream->close();
            delete stream;
            stream = NULL;
        }
    }
}

// Print out whatever layers we can find out from the Windows'
// registry and other environmental variables that may be used
// to point the Vulkan loader at a layer path.
ErrorResults PrintLayerInfo(void) {
    ErrorResults res = SUCCESSFUL;
    const char vulkan_reg_base[] = "SOFTWARE\\Khronos\\Vulkan";
    const char vulkan_reg_base_wow64[] = "SOFTWARE\\WOW6432Node\\Khronos\\Vulkan";
    char vulkan_impl_layer_reg_key[MAX_STRING_LENGTH];
    char cur_vulkan_layer_json[MAX_STRING_LENGTH];
    char generic_string[MAX_STRING_LENGTH];
    char full_layer_path[MAX_STRING_LENGTH];
    char env_value[MAX_STRING_LENGTH];
    uint32_t i = 0;
    uint32_t j = 0;
    FILE *fp = NULL;

// Dump implicit layer information first.
#if _WIN64 || __x86_64__ || __ppc64__
    snprintf(vulkan_impl_layer_reg_key, MAX_STRING_LENGTH - 1, "%s\\ImplicitLayers", vulkan_reg_base);
#else
    if (global_items.is_wow64) {
        snprintf(vulkan_impl_layer_reg_key, MAX_STRING_LENGTH - 1, "%s\\ImplicitLayers", vulkan_reg_base_wow64);
    } else {
        snprintf(vulkan_impl_layer_reg_key, MAX_STRING_LENGTH - 1, "%s\\ImplicitLayers", vulkan_reg_base);
    }
#endif

    PrintBeginTable("Implicit Layers", 4);
    PrintBeginTableRow();
    PrintTableElement("Layers in Registry");
    PrintTableElement(vulkan_impl_layer_reg_key);
    PrintTableElement("");
    PrintTableElement("");
    PrintEndTableRow();

    PrintImplicitLayersRegInfo(HKEY_LOCAL_MACHINE, vulkan_impl_layer_reg_key, cur_vulkan_layer_json, generic_string, res);
    PrintImplicitLayersRegInfo(HKEY_CURRENT_USER, vulkan_impl_layer_reg_key, cur_vulkan_layer_json, generic_string, res);

    PrintEndTable();

    // If the user's system has VK_LAYER_PATH set, dump out the layer
    // information found in that folder.  This is important because if
    // a user is having problems with the layers, they may be using
    // non-standard layers.
    if (0 != GetEnvironmentVariableA("VK_LAYER_PATH", env_value, MAX_STRING_LENGTH - 1)) {
        WIN32_FIND_DATAA ffd;
        HANDLE hFind;
        std::string cur_layer_path;
        bool keep_looping = false;
        uint32_t path = 0;

        PrintBeginTable("VK_LAYER_PATH Explicit Layers", 3);
        PrintBeginTableRow();
        PrintTableElement("VK_LAYER_PATH");
        PrintTableElement(env_value);
        PrintTableElement("");
        PrintEndTableRow();

        // VK_LAYER_PATH may have multiple folders listed in it (colon
        // ';' delimited)
        char *tok = strtok(env_value, ";");
        if (tok != NULL) {
            cur_layer_path = tok;
            keep_looping = true;
        } else {
            cur_layer_path = env_value;
        }

        do {
            if (keep_looping) {
                PrintBeginTableRow();
                sprintf(generic_string, "Path %d", path++);
                PrintTableElement(generic_string, ALIGN_CENTER);
                PrintTableElement(cur_layer_path);
                PrintTableElement("");
                PrintEndTableRow();
            }

            // Look for any JSON files in that folder.
            snprintf(full_layer_path, MAX_STRING_LENGTH - 1, "%s\\*.json", cur_layer_path.c_str());
            i = 0;
            hFind = FindFirstFileA(full_layer_path, &ffd);
            if (hFind != INVALID_HANDLE_VALUE) {
                do {
                    if (0 == (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                        snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", i++);
                        snprintf(cur_vulkan_layer_json, MAX_STRING_LENGTH - 1, "%s\\%s", cur_layer_path.c_str(), ffd.cFileName);

                        PrintBeginTableRow();
                        PrintTableElement(generic_string, ALIGN_RIGHT);
                        PrintTableElement(ffd.cFileName);
                        PrintTableElement("");
                        PrintEndTableRow();

                        std::ifstream *stream = NULL;
                        stream = new std::ifstream(cur_vulkan_layer_json, std::ifstream::in);
                        if (nullptr == stream || stream->fail()) {
                            PrintBeginTableRow();
                            PrintTableElement("");
                            PrintTableElement("ERROR reading JSON file!");
                            PrintTableElement("");
                            PrintEndTableRow();
                            res = MISSING_LAYER_JSON;
                        } else {
                            Json::Value root = Json::nullValue;
                            Json::Reader reader;
                            if (!reader.parse(*stream, root, false) || root.isNull()) {
                                // Report to the user the failure and their
                                // locations in the document.
                                PrintBeginTableRow();
                                PrintTableElement("");
                                PrintTableElement("ERROR parsing JSON file!");
                                PrintTableElement(reader.getFormattedErrorMessages());
                                PrintEndTableRow();
                                res = LAYER_JSON_PARSING_ERROR;
                            } else {
                                PrintExplicitLayerJsonInfo(cur_vulkan_layer_json, root, 3);
                            }

                            stream->close();
                            delete stream;
                            stream = NULL;
                        }
                    }
                } while (FindNextFileA(hFind, &ffd) != 0);

                FindClose(hFind);
            }

            tok = strtok(NULL, ";");
            if (tok == NULL) {
                keep_looping = false;
            } else {
                cur_layer_path = tok;
            }
        } while (keep_looping);

        PrintEndTable();
    }

    return res;
}

#elif __GNUC__

// Utility function to determine if a driver may exist in the folder.
bool CheckDriver(std::string &folder_loc, std::string &object_name) {
    bool success = false;
    std::string full_name = folder_loc;
    if (folder_loc.c_str()[folder_loc.size() - 1] != '/') {
        full_name += "/";
    }
    full_name += object_name;
    if (access(full_name.c_str(), R_OK) != -1) {
        success = true;
    }
    return success;
}

// Pointer to a function sed to validate if the system object is found
typedef bool (*PFN_CheckIfValid)(std::string &folder_loc, std::string &object_name);

bool FindLinuxSystemObject(std::string object_name, std::string &location, PFN_CheckIfValid func, bool break_on_first) {
    bool found_one = false;
    std::string path_to_check;
    char *env_value = getenv("LD_LIBRARY_PATH");

    for (uint32_t iii = 0; iii < 5; iii++) {
        switch (iii) {
            case 0:
                path_to_check = "/usr/lib";
                break;
            case 1:
#if __x86_64__ || __ppc64__
                path_to_check = "/usr/lib/x86_64-linux-gnu";
#else
                path_to_check = "/usr/lib/i386-linux-gnu";
#endif
                break;
            case 2:
#if __x86_64__ || __ppc64__
                path_to_check = "/usr/lib64";
#else
                path_to_check = "/usr/lib32";
#endif
                break;
            case 3:
                path_to_check = "/usr/local/lib";
                break;
            case 4:
#if __x86_64__ || __ppc64__
                path_to_check = "/usr/local/lib64";
#else
                path_to_check = "/usr/local/lib32";
#endif
                break;
            default:
                continue;
        }

        if (func(path_to_check, object_name)) {
            location = path_to_check + "/" + object_name;

            // We found one runtime, clear any failures
            found_one = true;
            if (break_on_first) {
                goto out;
            }
        }
    }

    // LD_LIBRARY_PATH may have multiple folders listed in it (colon
    // ':' delimited)
    if (env_value != NULL) {
        char *tok = strtok(env_value, ":");
        while (tok != NULL) {
            if (strlen(tok) > 0) {
                path_to_check = tok;
                if (func(path_to_check, object_name)) {
                    location = path_to_check + "/" + object_name;

                    // We found one runtime, clear any failures
                    found_one = true;
                }
            }
            tok = strtok(NULL, ":");
        }
    }

out:
    return found_one;
}

// Print out any information about the current system that we can
// capture to ease in debugging/investigation at a later time.
ErrorResults PrintSystemInfo(void) {
    ErrorResults res = SUCCESSFUL;
    FILE *fp;
    char path[1035];
    char generic_string[MAX_STRING_LENGTH];
    utsname buffer;
    struct statvfs fs_stats;
    int num_cpus;
    uint64_t memory;
    char *env_value;
    std::string cur_directory;
    std::string exe_directory;
    std::string desktop_session;

    BeginSection("System Info");

    // Environment section has information about the OS and the
    // execution environment.
    PrintBeginTable("Environment", 3);

    fp = popen("cat /etc/os-release", "r");
    if (fp == NULL) {
        PrintBeginTableRow();
        PrintTableElement("ERROR");
        PrintTableElement("Failed to cat /etc/os-release");
        PrintTableElement("");
        PrintEndTableRow();
        res = SYSTEM_CALL_FAILURE;
    } else {
        // Read the output a line at a time - output it.
        while (fgets(path, sizeof(path) - 1, fp) != NULL) {
            if (NULL != strstr(path, "PRETTY_NAME")) {
                uint32_t index;
                index = strlen(path) - 1;
                while (path[index] == ' ' || path[index] == '\t' || path[index] == '\r' || path[index] == '\n' ||
                       path[index] == '\"') {
                    path[index] = '\0';
                    index = strlen(path) - 1;
                }
                index = 13;
                while (path[index] == ' ' || path[index] == '\t' || path[index] == '\"') {
                    index++;
                }
                PrintBeginTableRow();
                PrintTableElement("Linux");
                PrintTableElement("");
                PrintTableElement("");
                PrintEndTableRow();
                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement("Distro");
                PrintTableElement(&path[index]);
                PrintEndTableRow();
                break;
            }
        }
        pclose(fp);
    }

    errno = 0;
    if (uname(&buffer) != 0) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("ERROR");
        PrintTableElement("Failed to query uname");
        PrintEndTableRow();
        res = SYSTEM_CALL_FAILURE;
    } else {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("Kernel Build");
        PrintTableElement(buffer.release);
        PrintEndTableRow();
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("Machine Target");
        PrintTableElement(buffer.machine);
        PrintEndTableRow();
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("Version");
        PrintTableElement(buffer.version);
        PrintEndTableRow();
    }

    env_value = getenv("DESKTOP_SESSION");
    if (env_value != NULL) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("DESKTOP_SESSION");
        PrintTableElement(env_value);
        PrintEndTableRow();

        desktop_session = env_value;
    }
    env_value = getenv("LD_LIBRARY_PATH");
    if (env_value != NULL) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("LD_LIBRARY_PATH");
        PrintTableElement(env_value);
        PrintEndTableRow();
    }
    env_value = getenv("GDK_BACKEND");
    if (env_value != NULL) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("GDK_BACKEND");
        PrintTableElement(env_value);
        PrintEndTableRow();
    }
    env_value = getenv("DISPLAY");
    if (env_value != NULL) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("DISPLAY");
        PrintTableElement(env_value);
        PrintEndTableRow();
    }
    env_value = getenv("WAYLAND_DISPLAY");
    if (env_value != NULL) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("WAYLAND_DISPLAY");
        PrintTableElement(env_value);
        PrintEndTableRow();
    }
    env_value = getenv("MIR_SOCKET");
    if (env_value != NULL) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("MIR_SOCKET");
        PrintTableElement(env_value);
        PrintEndTableRow();
    }

    PrintEndTable();

    if (getcwd(generic_string, MAX_STRING_LENGTH - 1) != NULL) {
        cur_directory = generic_string;
    } else {
        cur_directory = "";
    }

    // Output whatever generic hardware information we can find out about the
    // system.  Including how much memory and disk space is available.
    PrintBeginTable("Hardware", 3);

    num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    snprintf(generic_string, MAX_STRING_LENGTH - 1, "%d", num_cpus);

    PrintBeginTableRow();
    PrintTableElement("CPUs");
    PrintTableElement(generic_string);
    PrintTableElement("");
    PrintEndTableRow();

    memory = (sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE)) >> 10;
    if ((memory >> 10) > 0) {
        memory >>= 10;
        if ((memory >> 20) > 0) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u TB", static_cast<uint32_t>(memory >> 20));
        } else if ((memory >> 10) > 0) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u GB", static_cast<uint32_t>(memory >> 10));
        } else {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u MB", static_cast<uint32_t>(memory));
        }
    } else {
        snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u KB", static_cast<uint32_t>(memory));
    }
    PrintBeginTableRow();
    PrintTableElement("Memory");
    PrintTableElement("Physical");
    PrintTableElement(generic_string);
    PrintEndTableRow();

    // Print system disk space usage
    if (0 == statvfs("/etc/os-release", &fs_stats)) {
        uint64_t bytes_total = (uint64_t)fs_stats.f_bsize * (uint64_t)fs_stats.f_bavail;
        if ((bytes_total >> 40) > 0x0ULL) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u TB", static_cast<uint32_t>(bytes_total >> 40));
            PrintBeginTableRow();
            PrintTableElement("System Disk Space");
            PrintTableElement("Free");
            PrintTableElement(generic_string);
            PrintEndTableRow();
        } else if ((bytes_total >> 30) > 0x0ULL) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u GB", static_cast<uint32_t>(bytes_total >> 30));
            PrintBeginTableRow();
            PrintTableElement("System Disk Space");
            PrintTableElement("Free");
            PrintTableElement(generic_string);
        } else if ((bytes_total >> 20) > 0x0ULL) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u MB", static_cast<uint32_t>(bytes_total >> 20));
            PrintBeginTableRow();
            PrintTableElement("System Disk Space");
            PrintTableElement("Free");
            PrintTableElement(generic_string);
            PrintEndTableRow();
        } else if ((bytes_total >> 10) > 0x0ULL) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u KB", static_cast<uint32_t>(bytes_total >> 10));
            PrintBeginTableRow();
            PrintTableElement("System Disk Space");
            PrintTableElement("Free");
            PrintTableElement(generic_string);
            PrintEndTableRow();
        } else {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%u bytes", static_cast<uint32_t>(bytes_total));
            PrintBeginTableRow();
            PrintTableElement("System Disk Space");
            PrintTableElement("Free");
            PrintTableElement(generic_string);
            PrintEndTableRow();
        }
    }

    // Print current directory disk space info
    sprintf(generic_string, "df -h \'%s\' | awk \'{ print $4 } \' | tail -n 1", cur_directory.c_str());
    fp = popen(generic_string, "r");
    if (fp == NULL) {
        PrintBeginTableRow();
        PrintTableElement("Current Dir Disk Space");
        PrintTableElement("WARNING");
        PrintTableElement("Failed to determine current directory disk space");
        PrintEndTableRow();
    } else {
        PrintBeginTableRow();
        PrintTableElement("Current Dir Disk Space");
        PrintTableElement("Free");
        if (fgets(path, sizeof(path) - 1, fp) != NULL) {
            PrintTableElement(path);
        } else {
            PrintTableElement("Failed to determine current directory disk space");
        }
        PrintEndTableRow();
        pclose(fp);
    }
    PrintEndTable();

    // Print out information about this executable.
    PrintBeginTable("Executable", 2);

    PrintBeginTableRow();
    PrintTableElement("Exe Directory");
    PrintTableElement(global_items.exe_directory);
    PrintEndTableRow();

    PrintBeginTableRow();
    PrintTableElement("Current Directory");
    PrintTableElement(cur_directory);
    PrintEndTableRow();

    PrintBeginTableRow();
    PrintTableElement("App Version");
    PrintTableElement(APP_VERSION);
    PrintEndTableRow();

    uint32_t major = VK_VERSION_MAJOR(VK_API_VERSION_1_0);
    uint32_t minor = VK_VERSION_MINOR(VK_API_VERSION_1_0);
    uint32_t patch = VK_VERSION_PATCH(VK_HEADER_VERSION);
    snprintf(generic_string, MAX_STRING_LENGTH - 1, "%d.%d.%d", major, minor, patch);

    PrintBeginTableRow();
    PrintTableElement("Vulkan API Version");
    PrintTableElement(generic_string);
    PrintEndTableRow();

    PrintBeginTableRow();
    PrintTableElement("Byte Format");
#if __x86_64__ || __ppc64__
    PrintTableElement("64-bit");
#else
    PrintTableElement("32-bit");
#endif
    PrintEndTableRow();

    PrintEndTable();

    // Print out the rest of the useful system information.
    res = PrintDriverInfo();
    res = PrintRunTimeInfo();
    res = PrintSDKInfo();
    res = PrintLayerInfo();
    res = PrintLayerSettingsFileInfo();
    EndSection();

    return res;
}

bool VerifyOpen(std::string library_file, std::string &error) {
    bool success = false;
    void *handle = dlopen(library_file.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (NULL == handle) {
        error = dlerror();
    } else {
        dlclose(handle);
        success = true;
    }
    return success;
}

bool ReadDriverJson(std::string cur_driver_json, bool &found_lib) {
    bool found_json = false;
    std::ifstream *stream = NULL;
    Json::Value root = Json::nullValue;
    Json::Value inst_exts = Json::nullValue;
    Json::Value dev_exts = Json::nullValue;
    Json::Reader reader;
    char full_driver_path[MAX_STRING_LENGTH];
    char generic_string[MAX_STRING_LENGTH];
    uint32_t j = 0;

    stream = new std::ifstream(cur_driver_json.c_str(), std::ifstream::in);
    if (nullptr == stream || stream->fail()) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("Error reading JSON file");
        PrintTableElement(cur_driver_json);
        PrintEndTableRow();
        goto out;
    }

    if (!reader.parse(*stream, root, false) || root.isNull()) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("Error reading JSON file");
        PrintTableElement(reader.getFormattedErrorMessages());
        PrintEndTableRow();
        goto out;
    }

    PrintBeginTableRow();
    PrintTableElement("");
    PrintTableElement("JSON File Version");
    if (!root["file_format_version"].isNull()) {
        PrintTableElement(root["file_format_version"].asString());
    } else {
        PrintTableElement("MISSING!");
    }
    PrintEndTableRow();

    if (root["ICD"].isNull()) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("ICD Section");
        PrintTableElement("MISSING!");
        PrintEndTableRow();
        goto out;
    }

    found_json = true;

    PrintBeginTableRow();
    PrintTableElement("");
    PrintTableElement("API Version");
    if (!root["ICD"]["api_version"].isNull()) {
        PrintTableElement(root["ICD"]["api_version"].asString());
    } else {
        PrintTableElement("MISSING!");
    }
    PrintEndTableRow();

    PrintBeginTableRow();
    PrintTableElement("");
    PrintTableElement("Library Path");
    if (!root["ICD"]["library_path"].isNull()) {
        std::string driver_name = root["ICD"]["library_path"].asString();
        std::string location;
        bool could_load = true;
        std::string load_error;
        PrintTableElement(driver_name);
        PrintEndTableRow();

        if (GenerateLibraryPath(cur_driver_json.c_str(), driver_name.c_str(), MAX_STRING_LENGTH, full_driver_path)) {
            // First try the generated path.
            if (access(full_driver_path, R_OK) != -1) {
                found_lib = true;
                could_load = VerifyOpen(full_driver_path, load_error);
            } else if (driver_name.find("/") == std::string::npos) {
                if (FindLinuxSystemObject(driver_name, location, CheckDriver, true)) {
                    found_lib = true;
                    could_load = VerifyOpen(location, load_error);
                }
            }
        }
        if (!found_lib) {
            FILE *fp;
            sprintf(generic_string, "/sbin/ldconfig -v -N -p | grep %s | awk \'{ print $4 }\'", driver_name.c_str());
            fp = popen(generic_string, "r");
            if (fp == NULL) {
                snprintf(generic_string, MAX_STRING_LENGTH - 1,
                         "Failed to find driver %s "
                         "referenced by JSON %s",
                         driver_name.c_str(), cur_driver_json.c_str());
                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement("");
                PrintTableElement(generic_string);
                PrintEndTableRow();
            } else {
                char query_res[MAX_STRING_LENGTH];

                // Read the output a line at a time - output it.
                if (fgets(query_res, sizeof(query_res) - 1, fp) != NULL) {
                    sprintf(generic_string, "Found at %s", query_res);
                    PrintBeginTableRow();
                    PrintTableElement("");
                    PrintTableElement("");
                    PrintTableElement(generic_string);
                    PrintEndTableRow();
                    found_lib = true;
                    could_load = VerifyOpen(query_res, load_error);
                }
                fclose(fp);
            }
        } else if (!could_load) {
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("FAILED TO LOAD!");
            PrintTableElement(load_error);
            PrintEndTableRow();
        }
    } else {
        PrintTableElement("MISSING!");
        PrintEndTableRow();
    }

    char count_str[MAX_STRING_LENGTH];
    j = 0;
    dev_exts = root["ICD"]["device_extensions"];
    if (!dev_exts.isNull() && dev_exts.isArray()) {
        snprintf(count_str, MAX_STRING_LENGTH - 1, "%d", dev_exts.size());
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("Device Extensions");
        PrintTableElement(count_str);
        PrintEndTableRow();

        for (Json::ValueIterator dev_ext_it = dev_exts.begin(); dev_ext_it != dev_exts.end(); dev_ext_it++) {
            Json::Value dev_ext = (*dev_ext_it);
            Json::Value dev_ext_name = dev_ext["name"];
            if (!dev_ext_name.isNull()) {
                snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", j);

                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement(generic_string, ALIGN_RIGHT);
                PrintTableElement(dev_ext_name.asString());
                PrintEndTableRow();
            }
        }
    }
    inst_exts = root["ICD"]["instance_extensions"];
    j = 0;
    if (!inst_exts.isNull() && inst_exts.isArray()) {
        snprintf(count_str, MAX_STRING_LENGTH - 1, "%d", inst_exts.size());
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("Instance Extensions");
        PrintTableElement(count_str);
        PrintEndTableRow();

        for (Json::ValueIterator inst_ext_it =

                 inst_exts.begin();
             inst_ext_it != inst_exts.end(); inst_ext_it++) {
            Json::Value inst_ext = (*inst_ext_it);
            Json::Value inst_ext_name = inst_ext["name"];
            if (!inst_ext_name.isNull()) {
                snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", j);

                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement(generic_string, ALIGN_RIGHT);
                PrintTableElement(inst_ext_name.asString());
                PrintEndTableRow();
            }
        }
    }

out:

    if (nullptr != stream) {
        stream->close();
        delete stream;
        stream = NULL;
    }

    return found_json;
}

// Print out the information for every driver JSON in the appropriate
// system folders.
ErrorResults PrintDriverInfo(void) {
    ErrorResults res = SUCCESSFUL;
    bool found_json = false;
    bool found_lib = false;
    bool found_this_lib = false;
    uint32_t i = 0;
    char generic_string[MAX_STRING_LENGTH];
    char cur_vulkan_driver_json[MAX_STRING_LENGTH];
    char *home_env_value = NULL;
    char *drivers_env_value = NULL;
    char *icd_env_value = NULL;
    std::vector<std::string> driver_paths;
    int drivers_path_index = -1;

    PrintBeginTable("Vulkan Driver Info", 3);

    // There are several folders ICD JSONs could be in.  So,
    // try all of them.
    driver_paths.push_back("/etc/vulkan/icd.d");
    driver_paths.push_back("/usr/share/vulkan/icd.d");
    driver_paths.push_back("/usr/local/etc/vulkan/icd.d");
    driver_paths.push_back("/usr/local/share/vulkan/icd.d");

    home_env_value = getenv("HOME");
    if (NULL == home_env_value) {
        driver_paths.push_back("~/.local/share/vulkan/icd.d");
    } else {
        std::string home_icd_dir = home_env_value;
        home_icd_dir += "/.local/share/vulkan/icd.d";
        driver_paths.push_back(home_icd_dir);
    }

    // The user can override the drivers path manually
    drivers_env_value = getenv("VK_DRIVERS_PATH");
    if (NULL != drivers_env_value) {
        drivers_path_index = driver_paths.size();
        // VK_DRIVERS_PATH may have multiple folders listed in it (colon
        // ':' delimited)
        char *tok = strtok(drivers_env_value, ":");
        if (tok != NULL) {
            while (tok != NULL) {
                driver_paths.push_back(tok);
                tok = strtok(NULL, ":");
            }
        } else {
            driver_paths.push_back(drivers_env_value);
        }
    }

    // Loop through all folders discovered above.
    for (size_t dir = 0; dir < driver_paths.size(); dir++) {
        // Just to make things clear, make sure to add a
        // identifier before the drivers path results.
        if (dir == 0) {
            PrintBeginTableRow();
            PrintTableElement("Standard Paths");
            PrintTableElement("");
            PrintTableElement("");
            PrintEndTableRow();
        } else if (drivers_path_index >= 0 && dir == static_cast<size_t>(drivers_path_index)) {
            PrintBeginTableRow();
            PrintTableElement("VK_DRIVERS_PATH");
            PrintTableElement(drivers_env_value);
            PrintTableElement("");
            PrintEndTableRow();
        }

        // Make sure the directory exists.
        DIR *driver_dir = opendir(driver_paths[dir].c_str());
        if (NULL == driver_dir) {
            PrintBeginTableRow();
            PrintTableElement(driver_paths[dir], ALIGN_RIGHT);
            PrintTableElement("No such folder");
            PrintTableElement("");
            PrintEndTableRow();

            continue;
        }

        PrintBeginTableRow();
        PrintTableElement(driver_paths[dir], ALIGN_RIGHT);
        PrintTableElement("");
        PrintTableElement("");
        PrintEndTableRow();

        dirent *cur_ent;
        i = 0;
        while ((cur_ent = readdir(driver_dir)) != NULL) {
            if (NULL != strstr(cur_ent->d_name, ".json")) {
                snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", i++);
                snprintf(cur_vulkan_driver_json, MAX_STRING_LENGTH - 1, "%s/%s", driver_paths[dir].c_str(), cur_ent->d_name);

                PrintBeginTableRow();
                PrintTableElement(generic_string, ALIGN_RIGHT);
                PrintTableElement(cur_ent->d_name);
                PrintTableElement("");
                PrintEndTableRow();

                if (ReadDriverJson(cur_vulkan_driver_json, found_this_lib)) {
                    found_json = true;
                    found_lib |= found_this_lib;
                }
            }
        }
    }

    // The user can specify particularly what driver files to use
    icd_env_value = getenv("VK_ICD_FILENAMES");
    if (NULL != icd_env_value) {
        PrintBeginTableRow();
        PrintTableElement("VK_ICD_FILENAMES");
        PrintTableElement(icd_env_value);
        PrintTableElement("");
        PrintEndTableRow();

        // VK_ICD_FILENAMES may have multiple folders listed in it (colon
        // ':' delimited)
        char *tok = strtok(icd_env_value, ":");
        if (tok != NULL) {
            while (tok != NULL) {
                if (access(tok, R_OK) != -1) {
                    PrintBeginTableRow();
                    PrintTableElement(tok, ALIGN_RIGHT);
                    PrintTableElement("");
                    PrintTableElement("");
                    PrintEndTableRow();
                    if (ReadDriverJson(tok, found_this_lib)) {
                        found_json = true;
                        found_lib |= found_this_lib;
                    }
                } else {
                    PrintBeginTableRow();
                    PrintTableElement(tok, ALIGN_RIGHT);
                    PrintTableElement("No such file");
                    PrintTableElement("");
                    PrintEndTableRow();
                }
                tok = strtok(NULL, ":");
            }
        } else {
            if (access(icd_env_value, R_OK) != -1) {
                PrintBeginTableRow();
                PrintTableElement(icd_env_value, ALIGN_RIGHT);
                PrintTableElement("");
                PrintTableElement("");
                PrintEndTableRow();
                if (ReadDriverJson(icd_env_value, found_this_lib)) {
                    found_json = true;
                    found_lib |= found_this_lib;
                }
            } else {
                PrintBeginTableRow();
                PrintTableElement(icd_env_value, ALIGN_RIGHT);
                PrintTableElement("No such file");
                PrintTableElement("");
                PrintEndTableRow();
            }
        }
    }

    PrintEndTable();

    if (!found_json) {
        res = MISSING_DRIVER_JSON;
    } else if (!found_lib) {
        res = MISSING_DRIVER_LIB;
    }

    return res;
}

// Print out all the runtime files found in a given location.  This way we
// capture the full state of the system.
ErrorResults PrintRuntimesInFolder(std::string &folder_loc, std::string &object_name, bool print_header = true) {
    DIR *runtime_dir;
    ErrorResults res = SUCCESSFUL;

    runtime_dir = opendir(folder_loc.c_str());
    if (NULL != runtime_dir) {
        bool file_found = false;
        FILE *pfp;
        uint32_t i = 0;
        dirent *cur_ent;
        std::string command_str;
        std::stringstream generic_str;
        char path[1035];

        if (print_header) {
            PrintBeginTableRow();
            PrintTableElement(folder_loc, ALIGN_RIGHT);
            PrintTableElement("");
            PrintTableElement("");
            PrintEndTableRow();
        }

        while ((cur_ent = readdir(runtime_dir)) != NULL) {
            if (NULL != strstr(cur_ent->d_name, object_name.c_str()) && strlen(cur_ent->d_name) == 14) {
                // Get the source of this symbolic link
                command_str = "stat -c%N \'";
                command_str += folder_loc;
                command_str += "/";
                command_str += cur_ent->d_name;
                command_str += "\'";
                pfp = popen(command_str.c_str(), "r");

                generic_str << "[" << i++ << "]";

                PrintBeginTableRow();
                PrintTableElement(generic_str.str(), ALIGN_RIGHT);

                file_found = true;

                if (pfp == NULL) {
                    PrintTableElement(cur_ent->d_name);
                    PrintTableElement("Failed to retrieve symbolic link");
                    res = SYSTEM_CALL_FAILURE;
                } else {
                    if (NULL != fgets(path, sizeof(path) - 1, pfp)) {
                        std::string cmd = path;
                        size_t arrow_loc = cmd.find("->");
                        if (arrow_loc == std::string::npos) {
                            std::string trimmed_path = TrimWhitespace(path, " \t\n\r\'\"");

                            PrintTableElement(trimmed_path);
                            PrintTableElement("");
                        } else {
                            std::string before_arrow = cmd.substr(0, arrow_loc);
                            std::string trim_before = TrimWhitespace(before_arrow, " \t\n\r\'\"");
                            std::string after_arrow = cmd.substr(arrow_loc + 2, std::string::npos);
                            std::string trim_after = TrimWhitespace(after_arrow, " \t\n\r\'\"");
                            PrintTableElement(trim_before);
                            PrintTableElement(trim_after);
                        }
                    } else {
                        PrintTableElement(cur_ent->d_name);
                        PrintTableElement("Failed to retrieve symbolic link");
                    }

                    PrintEndTableRow();

                    pclose(pfp);
                }
            }
        }
        if (!file_found) {
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("No libvulkan.so files found");
            PrintTableElement("");
            PrintEndTableRow();
        }
        closedir(runtime_dir);
    } else {
        PrintBeginTableRow();
        PrintTableElement(folder_loc, ALIGN_RIGHT);
        PrintTableElement("No such folder");
        PrintTableElement("");
        PrintEndTableRow();
    }

    return res;
}

// Utility function to determine if a runtime exists in the folder
bool CheckRuntime(std::string &folder_loc, std::string &object_name) {
    return (SUCCESSFUL == PrintRuntimesInFolder(folder_loc, object_name));
}

// Print out whatever Vulkan runtime information we can gather from the
// standard system paths, etc.
ErrorResults PrintRunTimeInfo(void) {
    ErrorResults res = SUCCESSFUL;
    const char vulkan_so_prefix[] = "libvulkan.so.";
    char path[1035];
    char generic_string[MAX_STRING_LENGTH];
    char buff[PATH_MAX];
    std::string runtime_dir_name;
    std::string location;
    FILE *pfp;
    PrintBeginTable("Vulkan Runtimes", 3);

    PrintBeginTableRow();
    PrintTableElement("Possible Runtime Folders");
    PrintTableElement("");
    PrintTableElement("");
    PrintEndTableRow();

    if (!FindLinuxSystemObject(vulkan_so_prefix, location, CheckRuntime, false)) {
        res = VULKAN_CANT_FIND_RUNTIME;
    }

    ssize_t len = ::readlink("/proc/self/exe", buff, sizeof(buff) - 1);
    if (len != -1) {
        buff[len] = '\0';

        std::string runtime_dir_id = "Runtime Folder Used By via";
        snprintf(generic_string, MAX_STRING_LENGTH - 1, "ldd \'%s\'", buff);
        pfp = popen(generic_string, "r");
        if (pfp == NULL) {
            PrintBeginTableRow();
            PrintTableElement(runtime_dir_id);
            PrintTableElement("Failed to query via library info");
            PrintTableElement("");
            PrintEndTableRow();
            res = SYSTEM_CALL_FAILURE;
        } else {
            bool found = false;
            while (fgets(path, sizeof(path) - 1, pfp) != NULL) {
                if (NULL != strstr(path, vulkan_so_prefix)) {
                    std::string cmd = path;
                    size_t arrow_loc = cmd.find("=>");
                    if (arrow_loc == std::string::npos) {
                        std::string trimmed_path = TrimWhitespace(path, " \t\n\r\'\"");
                        PrintBeginTableRow();
                        PrintTableElement(runtime_dir_id);
                        PrintTableElement(trimmed_path);
                        PrintTableElement("");
                        PrintEndTableRow();
                    } else {
                        std::string after_arrow = cmd.substr(arrow_loc + 2);
                        std::string before_slash = after_arrow.substr(0, after_arrow.rfind("/"));
                        std::string trimmed = TrimWhitespace(before_slash, " \t\n\r\'\"");

                        PrintBeginTableRow();
                        PrintTableElement(runtime_dir_id);
                        PrintTableElement(trimmed);
                        PrintTableElement("");
                        PrintEndTableRow();

                        std::string find_so = vulkan_so_prefix;
                        ErrorResults temp_res = PrintRuntimesInFolder(trimmed, find_so, false);
                        if (!found) {
                            res = temp_res;
                        } else {
                            // We found one runtime, clear any failures
                            if (res == VULKAN_CANT_FIND_RUNTIME) {
                                res = SUCCESSFUL;
                                found = true;
                            }
                        }
                    }
                    break;
                }
            }
            if (!found) {
                PrintBeginTableRow();
                PrintTableElement(runtime_dir_id);
                PrintTableElement("Failed to find Vulkan SO used for via");
                PrintTableElement("");
                PrintEndTableRow();
            }
            pclose(pfp);
        }
        PrintEndTableRow();
    }

    PrintEndTable();

    return res;
}

// Print out the explicit layers that are stored in any of the standard
// locations.
ErrorResults PrintExplicitLayersInFolder(std::string &id, std::string &folder_loc) {
    ErrorResults res = SUCCESSFUL;
    DIR *layer_dir;

    layer_dir = opendir(folder_loc.c_str());
    if (NULL != layer_dir) {
        dirent *cur_ent;
        std::string cur_layer;
        char generic_string[MAX_STRING_LENGTH];
        uint32_t i = 0;
        bool found_json = false;

        PrintBeginTableRow();
        PrintTableElement(id, ALIGN_RIGHT);
        PrintTableElement(folder_loc);
        PrintTableElement("");
        PrintEndTableRow();

        // Loop through each JSON in a given folder
        while ((cur_ent = readdir(layer_dir)) != NULL) {
            if (NULL != strstr(cur_ent->d_name, ".json")) {
                found_json = true;

                snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", i++);
                cur_layer = folder_loc;
                cur_layer += "/";
                cur_layer += cur_ent->d_name;

                // Parse the JSON file
                std::ifstream *stream = NULL;
                stream = new std::ifstream(cur_layer, std::ifstream::in);
                if (nullptr == stream || stream->fail()) {
                    PrintBeginTableRow();
                    PrintTableElement(generic_string, ALIGN_RIGHT);
                    PrintTableElement(cur_ent->d_name);
                    PrintTableElement("ERROR reading JSON file!");
                    PrintEndTableRow();
                    res = MISSING_LAYER_JSON;
                } else {
                    Json::Value root = Json::nullValue;
                    Json::Reader reader;
                    if (!reader.parse(*stream, root, false) || root.isNull()) {
                        // Report to the user the failure and their
                        // locations in the document.
                        PrintBeginTableRow();
                        PrintTableElement(generic_string, ALIGN_RIGHT);
                        PrintTableElement(cur_ent->d_name);
                        PrintTableElement(reader.getFormattedErrorMessages());
                        PrintEndTableRow();
                        res = LAYER_JSON_PARSING_ERROR;
                    } else {
                        PrintBeginTableRow();
                        PrintTableElement(generic_string, ALIGN_RIGHT);
                        PrintTableElement(cur_ent->d_name);
                        PrintTableElement("");
                        PrintEndTableRow();

                        // Dump out the standard explicit layer information.
                        PrintExplicitLayerJsonInfo(cur_layer.c_str(), root, 3);
                    }

                    stream->close();
                    delete stream;
                    stream = NULL;
                }
            }
        }
        if (!found_json) {
            PrintBeginTableRow();
            PrintTableElement(id, ALIGN_RIGHT);
            PrintTableElement(folder_loc);
            PrintTableElement("No JSON files found");
            PrintEndTableRow();
        }
        closedir(layer_dir);
    } else {
        PrintBeginTableRow();
        PrintTableElement(id, ALIGN_RIGHT);
        PrintTableElement(folder_loc);
        PrintTableElement("No such folder");
        PrintEndTableRow();
    }

    return res;
}

// Print out information on whatever LunarG Vulkan SDKs we can find on
// the system using the standard locations and environmental variables.
// This includes listing what layers are available from the SDK.
ErrorResults PrintSDKInfo(void) {
    ErrorResults res = SUCCESSFUL;
    bool sdk_exists = false;
    std::string sdk_path;
    std::string sdk_env_name;
    const char vulkan_so_prefix[] = "libvulkan.so.";
    DIR *sdk_dir;
    dirent *cur_ent;
    char *env_value;

    PrintBeginTable("LunarG Vulkan SDKs", 3);

    for (uint32_t dir = 0; dir < 2; dir++) {
        switch (dir) {
            case 0:
                sdk_env_name = "VK_SDK_PATH";
                env_value = getenv(sdk_env_name.c_str());
                if (env_value == NULL) {
                    continue;
                }
                sdk_path = env_value;
                break;
            case 1:
                sdk_env_name = "VULKAN_SDK";
                env_value = getenv(sdk_env_name.c_str());
                if (env_value == NULL) {
                    continue;
                }
                sdk_path = env_value;
                break;
            default:
                res = UNKNOWN_ERROR;
                continue;
        }

        std::string explicit_layer_path = sdk_path;
        explicit_layer_path += "/etc/explicit_layer.d";

        sdk_dir = opendir(explicit_layer_path.c_str());
        if (NULL != sdk_dir) {
            while ((cur_ent = readdir(sdk_dir)) != NULL) {
                if (NULL != strstr(cur_ent->d_name, vulkan_so_prefix) && strlen(cur_ent->d_name) == 14) {
                }
            }
            closedir(sdk_dir);

            res = PrintExplicitLayersInFolder(sdk_env_name, explicit_layer_path);

            global_items.sdk_found = true;
            global_items.sdk_path = sdk_path;
            sdk_exists = true;
        }
    }

    if (!sdk_exists) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("No installed SDKs found");
        PrintTableElement("");
        PrintEndTableRow();
    }

    PrintEndTable();

    return res;
}

// Print out whatever layers we can find out from other environmental
// variables that may be used to point the Vulkan loader at a layer path.
ErrorResults PrintLayerInfo(void) {
    ErrorResults res = SUCCESSFUL;
    uint32_t i = 0;
    char generic_string[MAX_STRING_LENGTH];
    char cur_vulkan_layer_json[MAX_STRING_LENGTH];
    DIR *layer_dir;
    dirent *cur_ent;
    std::string layer_path;
    char *env_value = NULL;

    // Dump out implicit layer information first
    PrintBeginTable("Implicit Layers", 3);

    // There are several folders implicit layers could be in.  So,
    // try all of them.
    for (uint32_t dir = 0; dir < 5; dir++) {
        std::string cur_layer_path;
        switch (dir) {
            case 0:
                cur_layer_path = "/etc/vulkan/implicit_layer.d";
                break;
            case 1:
                cur_layer_path = "/usr/share/vulkan/implicit_layer.d";
                break;
            case 2:
                cur_layer_path = "/usr/local/etc/vulkan/implicit_layer.d";
                break;
            case 3:
                cur_layer_path = "/usr/local/share/vulkan/implicit_layer.d";
                break;
            case 4:
                env_value = getenv("HOME");
                if (NULL == env_value) {
                    cur_layer_path = "~/.local/share/vulkan/implicit_layer.d";
                } else {
                    cur_layer_path = env_value;
                    cur_layer_path += "/.local/share/vulkan/implicit_layer.d";
                }
                break;
            default:
                continue;
        }

        layer_dir = opendir(cur_layer_path.c_str());
        if (NULL != layer_dir) {
            PrintBeginTableRow();
            PrintTableElement(cur_layer_path, ALIGN_RIGHT);
            PrintTableElement("");
            PrintTableElement("");
            PrintEndTableRow();
            while ((cur_ent = readdir(layer_dir)) != NULL) {
                if (NULL != strstr(cur_ent->d_name, ".json")) {
                    snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", i++);
                    snprintf(cur_vulkan_layer_json, MAX_STRING_LENGTH - 1, "%s/%s", cur_layer_path.c_str(), cur_ent->d_name);

                    PrintBeginTableRow();
                    PrintTableElement(generic_string, ALIGN_RIGHT);
                    PrintTableElement(cur_ent->d_name);
                    PrintTableElement("");
                    PrintEndTableRow();

                    std::ifstream *stream = NULL;
                    stream = new std::ifstream(cur_vulkan_layer_json, std::ifstream::in);
                    if (nullptr == stream || stream->fail()) {
                        PrintBeginTableRow();
                        PrintTableElement("");
                        PrintTableElement("ERROR reading JSON file!");
                        PrintTableElement("");
                        PrintEndTableRow();
                        res = MISSING_LAYER_JSON;
                    } else {
                        Json::Value root = Json::nullValue;
                        Json::Reader reader;
                        if (!reader.parse(*stream, root, false) || root.isNull()) {
                            // Report to the user the failure and their
                            // locations in the document.
                            PrintBeginTableRow();
                            PrintTableElement("");
                            PrintTableElement("ERROR parsing JSON file!");
                            PrintTableElement(reader.getFormattedErrorMessages());
                            PrintEndTableRow();
                            res = LAYER_JSON_PARSING_ERROR;
                        } else {
                            PrintExplicitLayerJsonInfo(cur_vulkan_layer_json, root, 3);
                        }

                        stream->close();
                        delete stream;
                        stream = NULL;
                    }
                }
            }
            closedir(layer_dir);
        } else {
            PrintBeginTableRow();
            PrintTableElement(cur_layer_path, ALIGN_RIGHT);
            PrintTableElement("Directory does not exist");
            PrintTableElement("");
            PrintEndTableRow();
        }
    }
    PrintEndTable();

    // Dump out any explicit layer information.
    PrintBeginTable("Explicit Layers", 3);

    PrintBeginTableRow();
    PrintTableElement("Standard Paths");
    PrintTableElement("");
    PrintTableElement("");
    PrintEndTableRow();

    // There are several folders explicit layers could be in.  So,
    // try all of them.
    for (uint32_t dir = 0; dir < 5; dir++) {
        std::string cur_layer_path;
        std::string explicit_layer_id;
        std::string explicit_layer_path = cur_layer_path;
        char *env_value = NULL;
        switch (dir) {
            case 0:
                cur_layer_path = "/etc/vulkan/explicit_layer.d";
                explicit_layer_id = "/etc/vulkan";
                break;
            case 1:
                cur_layer_path = "/usr/share/vulkan/explicit_layer.d";
                explicit_layer_id = "/usr/share/vulkan";
                break;
            case 2:
                cur_layer_path = "/usr/local/etc/vulkan/explicit_layer.d";
                explicit_layer_id = "/usr/local/etc/vulkan";
                break;
            case 3:
                cur_layer_path = "/usr/local/share/vulkan/explicit_layer.d";
                explicit_layer_id = "/usr/local/share/vulkan";
                break;
            case 4:
                explicit_layer_id = "$HOME/.local/share/vulkan/explicit_layer.d";
                env_value = getenv("HOME");
                if (NULL == env_value) {
                    cur_layer_path = "~/.local/share/vulkan/explicit_layer.d";
                } else {
                    cur_layer_path = env_value;
                    cur_layer_path += "/.local/share/vulkan/explicit_layer.d";
                }
                break;
            default:
                continue;
        }

        res = PrintExplicitLayersInFolder(explicit_layer_id, cur_layer_path);
    }

    // Look at the VK_LAYER_PATH environment variable paths if it is set.
    env_value = getenv("VK_LAYER_PATH");
    std::string cur_json;
    if (NULL != env_value) {
        char *tok = strtok(env_value, ":");
        std::string explicit_layer_id = "VK_LAYER_PATH";

        PrintBeginTableRow();
        PrintTableElement("VK_LAYER_PATH");
        PrintTableElement("");
        PrintTableElement("");
        PrintEndTableRow();

        if (NULL != tok) {
            uint32_t offset = 0;
            std::stringstream cur_name;
            while (NULL != tok) {
                cur_json = tok;
                cur_name.str("");
                cur_name << "Path " << offset++;
                explicit_layer_id = cur_name.str();
                res = PrintExplicitLayersInFolder(explicit_layer_id, cur_json);
                tok = strtok(NULL, ":");
            }
        } else {
            cur_json = env_value;
            res = PrintExplicitLayersInFolder(explicit_layer_id, cur_json);
        }
    }

    PrintEndTable();

    return res;
}

// Run the test in the specified directory with the corresponding
// command-line arguments.
// Returns 0 on no error, 1 if test file wasn't found, and -1
// on any other errors.
int RunTestInDirectory(std::string path, std::string test, std::string cmd_line) {
    char orig_dir[MAX_STRING_LENGTH];
    int err_code = -1;
    orig_dir[0] = '\0';
    if (NULL != getcwd(orig_dir, MAX_STRING_LENGTH - 1)) {
        int err = chdir(path.c_str());
        if (-1 != err) {
            if (-1 != access(test.c_str(), X_OK)) {
                printf("cmd_line - %s\n", cmd_line.c_str());
                err_code = system(cmd_line.c_str());
            } else {
                // Can't run because it's either not there or an actual
                // exe.  So, just return a separate error code.
                err_code = 1;
            }
        } else {
            // Path doesn't exist at all
            err_code = 1;
        }
        chdir(orig_dir);
    }
    return err_code;
}

#endif

// Following functions should be OS agnostic:
//==========================================

// Trim any whitespace preceeding or following the actual
// content inside of a string.  The actual items labeled
// as whitespace are passed in as the second set of
// parameters.
std::string TrimWhitespace(const std::string &str, const std::string &whitespace) {
    const auto strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos) {
        return "";  // no content
    }

    const auto strEnd = str.find_last_not_of(whitespace);
    const auto strRange = strEnd - strBegin + 1;

    return str.substr(strBegin, strRange);
}

// Print any information found on the current vk_layer_settings.txt
// file being used.  It looks in the current folder first, and then will
// look in any defined by the registry variable VK_LAYER_SETTINGS_PATH.
ErrorResults PrintLayerSettingsFileInfo(void) {
    ErrorResults res = SUCCESSFUL;
    char *settings_path = NULL;
    std::string settings_file;
    std::map<std::string, std::vector<SettingPair>> settings;

    PrintBeginTable("Layer Settings File", 4);

// If the settings path environment variable is set, use that.
#ifdef _WIN32
    char generic_string[MAX_STRING_LENGTH];
    if (0 != GetEnvironmentVariableA("VK_LAYER_SETTINGS_PATH", generic_string, MAX_STRING_LENGTH - 1)) {
        settings_path = generic_string;
        settings_file = settings_path;
        settings_file += '\\';
    }
#else
    settings_path = getenv("VK_LAYER_SETTINGS_PATH");
    if (NULL != settings_path) {
        settings_file = settings_path;
        settings_file += '/';
    }
#endif
    settings_file += "vk_layer_settings.txt";

    PrintBeginTableRow();
    PrintTableElement("VK_LAYER_SETTINGS_PATH");
    if (NULL != settings_path) {
        PrintTableElement(settings_path);
    } else {
        PrintTableElement("Not Defined");
    }
    PrintTableElement("");
    PrintTableElement("");
    PrintEndTableRow();

    // Load the file from the appropriate location
    PrintBeginTableRow();
    PrintTableElement("Settings File");
    PrintTableElement("vk_layer_settings.txt");
    std::ifstream *settings_stream = new std::ifstream(settings_file, std::ifstream::in);
    if (nullptr == settings_stream || settings_stream->fail()) {
        // No file was found.  This is NOT an error.
        PrintTableElement("Not Found");
        PrintTableElement("");
        PrintEndTableRow();
    } else {
        // We found a file, so parse it.
        PrintTableElement("Found");
        PrintTableElement("");
        PrintEndTableRow();

        // The settings file is a text file where:
        //  - # indicates a comment
        //  - Settings are stored in the fasion:
        //        <layer_name>.<setting> = <value>
        while (settings_stream->good()) {
            std::string cur_line;
            getline(*settings_stream, cur_line);
            std::string trimmed_line = TrimWhitespace(cur_line);

            // Skip blank and comment lines
            if (trimmed_line.length() == 0 || trimmed_line.c_str()[0] == '#') {
                continue;
            }

            // If no equal, treat as unknown
            size_t equal_loc = trimmed_line.find("=");
            if (equal_loc == std::string::npos) {
                continue;
            }

            SettingPair new_pair;

            std::string before_equal = trimmed_line.substr(0, equal_loc);
            std::string after_equal = trimmed_line.substr(equal_loc + 1, std::string::npos);
            new_pair.value = TrimWhitespace(after_equal);

            std::string trimmed_setting = TrimWhitespace(before_equal);

            // Look for period
            std::string setting_layer = "--None--";
            std::string setting_name = "";
            size_t period_loc = trimmed_setting.find(".");
            if (period_loc == std::string::npos) {
                setting_name = trimmed_setting;
            } else {
                setting_layer = trimmed_setting.substr(0, period_loc);
                setting_name = trimmed_setting.substr(period_loc + 1, std::string::npos);
            }
            new_pair.name = setting_name;

            // Add items to settings map for now
            if (settings.find(setting_layer) == settings.end()) {
                // Not found
                std::vector<SettingPair> new_vector;
                new_vector.push_back(new_pair);
                settings[setting_layer] = new_vector;
            } else {
                // Already exists
                std::vector<SettingPair> &cur_vector = settings[setting_layer];
                cur_vector.push_back(new_pair);
            }
        }

        // Now that all items have been grouped in the settings map
        // appropriately, print
        // them out
        for (auto layer_iter = settings.begin(); layer_iter != settings.end(); layer_iter++) {
            std::vector<SettingPair> &cur_vector = layer_iter->second;
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement(layer_iter->first, ALIGN_RIGHT);
            PrintTableElement("");
            PrintTableElement("");
            PrintEndTableRow();
            for (uint32_t cur_item = 0; cur_item < cur_vector.size(); cur_item++) {
                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement("");
                PrintTableElement(cur_vector[cur_item].name);
                PrintTableElement(cur_vector[cur_item].value);
                PrintEndTableRow();
            }
        }

        settings_stream->close();
        delete settings_stream;
    }
    PrintEndTable();

    return res;
}

// Print out the information stored in an explicit layer's JSON file.
void PrintExplicitLayerJsonInfo(const char *layer_json_filename, Json::Value root, uint32_t num_cols) {
    char generic_string[MAX_STRING_LENGTH];
    uint32_t cur_col;
    uint32_t ext;
    if (!root["layer"].isNull()) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("Name");
        if (!root["layer"]["name"].isNull()) {
            PrintTableElement(root["layer"]["name"].asString());
        } else {
            PrintTableElement("MISSING!");
        }
        cur_col = 3;
        while (num_cols > cur_col) {
            PrintTableElement("");
            cur_col++;
        }
        PrintEndTableRow();

        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("Description");
        if (!root["layer"]["description"].isNull()) {
            PrintTableElement(root["layer"]["description"].asString());
        } else {
            PrintTableElement("MISSING!");
        }
        cur_col = 3;
        while (num_cols > cur_col) {
            PrintTableElement("");
            cur_col++;
        }
        PrintEndTableRow();

        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("API Version");
        if (!root["layer"]["api_version"].isNull()) {
            PrintTableElement(root["layer"]["api_version"].asString());
        } else {
            PrintTableElement("MISSING!");
        }
        cur_col = 3;
        while (num_cols > cur_col) {
            PrintTableElement("");
            cur_col++;
        }
        PrintEndTableRow();

        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("JSON File Version");
        if (!root["file_format_version"].isNull()) {
            PrintTableElement(root["file_format_version"].asString());
        } else {
            PrintTableElement("MISSING!");
        }
        cur_col = 3;
        while (num_cols > cur_col) {
            PrintTableElement("");
            cur_col++;
        }
        PrintEndTableRow();

        Json::Value component_layers = root["layer"]["component_layers"];
        Json::Value library_path = root["layer"]["library_path"];
        if (!component_layers.isNull() && !library_path.isNull()) {
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Library Path / Component Layers");
            PrintTableElement("BOTH DEFINED!");
            cur_col = 3;
            while (num_cols > cur_col) {
                PrintTableElement("");
                cur_col++;
            }
            PrintEndTableRow();
        } else if (!library_path.isNull()) {
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Library Path");
            PrintTableElement(library_path.asString());
            cur_col = 3;
            while (num_cols > cur_col) {
                PrintTableElement("");
                cur_col++;
            }
            PrintEndTableRow();

#ifdef _WIN32
            // On Windows, we can query the file version, so do so.
            char full_layer_path[MAX_STRING_LENGTH];
            if (GenerateLibraryPath(layer_json_filename, library_path.asString().c_str(), MAX_STRING_LENGTH,
                                    full_layer_path) &&
                GetFileVersion(full_layer_path, MAX_STRING_LENGTH, generic_string)) {
                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement("Layer File Version");
                PrintTableElement(generic_string);
                cur_col = 3;
                while (num_cols > cur_col) {
                    PrintTableElement("");
                    cur_col++;
                }
                PrintEndTableRow();
            }
#endif

            char count_str[MAX_STRING_LENGTH];
            Json::Value dev_exts = root["layer"]["device_extensions"];
            ext = 0;
            if (!dev_exts.isNull() && dev_exts.isArray()) {
                snprintf(count_str, MAX_STRING_LENGTH - 1, "%d", dev_exts.size());
                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement("Device Extensions");
                PrintTableElement(count_str);
                cur_col = 3;
                while (num_cols > cur_col) {
                    PrintTableElement("");
                    cur_col++;
                }
                PrintEndTableRow();

                for (Json::ValueIterator dev_ext_it = dev_exts.begin(); dev_ext_it != dev_exts.end(); dev_ext_it++) {
                    Json::Value dev_ext = (*dev_ext_it);
                    Json::Value dev_ext_name = dev_ext["name"];
                    if (!dev_ext_name.isNull()) {
                        snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", ext);
                        PrintBeginTableRow();
                        PrintTableElement("");
                        PrintTableElement(generic_string, ALIGN_RIGHT);
                        PrintTableElement(dev_ext_name.asString());
                        cur_col = 3;
                        while (num_cols > cur_col) {
                            PrintTableElement("");
                            cur_col++;
                        }
                        PrintEndTableRow();
                    }
                }
            }
            Json::Value inst_exts = root["layer"]["instance_extensions"];
            ext = 0;
            if (!inst_exts.isNull() && inst_exts.isArray()) {
                snprintf(count_str, MAX_STRING_LENGTH - 1, "%d", inst_exts.size());
                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement("Instance Extensions");
                PrintTableElement(count_str);
                cur_col = 3;
                while (num_cols > cur_col) {
                    PrintTableElement("");
                    cur_col++;
                }
                PrintEndTableRow();

                for (Json::ValueIterator inst_ext_it = inst_exts.begin(); inst_ext_it != inst_exts.end(); inst_ext_it++) {
                    Json::Value inst_ext = (*inst_ext_it);
                    Json::Value inst_ext_name = inst_ext["name"];
                    if (!inst_ext_name.isNull()) {
                        snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", ext);
                        PrintBeginTableRow();
                        PrintTableElement("");
                        PrintTableElement(generic_string, ALIGN_RIGHT);
                        PrintTableElement(inst_ext_name.asString());
                        cur_col = 3;
                        while (num_cols > cur_col) {
                            PrintTableElement("");
                            cur_col++;
                        }
                        PrintEndTableRow();
                    }
                }
            }
        } else if (!component_layers.isNull()) {
            if (component_layers.isArray()) {
                snprintf(generic_string, MAX_STRING_LENGTH - 1, "%d", component_layers.size());
                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement("Component Layers");
                PrintTableElement(generic_string);
                PrintEndTableRow();

                for (Json::ValueIterator cl_it = component_layers.begin(); cl_it != component_layers.end(); cl_it++) {
                    Json::Value comp_layer = (*cl_it);
                    PrintBeginTableRow();
                    PrintTableElement("");
                    PrintTableElement("");
                    PrintTableElement(comp_layer.asString(), ALIGN_RIGHT);
                    PrintEndTableRow();
                }
            } else {
                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement("Component Layers");
                PrintTableElement("NOT AN ARRAY!");
                cur_col = 3;
                while (num_cols > cur_col) {
                    PrintTableElement("");
                    cur_col++;
                }
                PrintEndTableRow();
            }
        } else {
            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Library Path / Component Layers");
            PrintTableElement("MISSING!");
            cur_col = 3;
            while (num_cols > cur_col) {
                PrintTableElement("");
                cur_col++;
            }
            PrintEndTableRow();
        }

    } else {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("Layer Section");
        PrintTableElement("MISSING!");
        cur_col = 3;
        while (num_cols > cur_col) {
            PrintTableElement("");
            cur_col++;
        }
        PrintEndTableRow();
    }
}

// Print out the information about an Implicit layer stored in
// it's JSON file.  For the most part, it is similar to an
// explicit layer, so we re-use that code.  However, implicit
// layers have a DISABLE environment variable that can be used
// to disable the layer by default.  Additionally, some implicit
// layers have an ENABLE environment variable so that they are
// disabled by default, but can be enabled.
void PrintImplicitLayerJsonInfo(const char *layer_json_filename, Json::Value root) {
    bool enabled = true;
    std::string enable_env_variable = "--NONE--";
    bool enable_var_set = false;
    char enable_env_value[16];
    std::string disable_env_variable = "--NONE--";
    bool disable_var_set = false;
    char disable_env_value[16];

    PrintExplicitLayerJsonInfo(layer_json_filename, root, 4);

    Json::Value enable = root["layer"]["enable_environment"];
    if (!enable.isNull()) {
        for (Json::Value::iterator en_iter = enable.begin(); en_iter != enable.end(); en_iter++) {
            if (en_iter.key().isNull()) {
                continue;
            }
            enable_env_variable = en_iter.key().asString();
            // If an enable define exists, set it to disabled by default.
            enabled = false;
#ifdef _WIN32
            if (0 != GetEnvironmentVariableA(enable_env_variable.c_str(), enable_env_value, 15)) {
#else
            char *enable_env = getenv(enable_env_variable.c_str());
            if (NULL != enable_env) {
                strncpy(enable_env_value, enable_env, 15);
                enable_env_value[15] = '\0';
#endif
                if (atoi(enable_env_value) != 0) {
                    enable_var_set = true;
                    enabled = true;
                }
            }
            break;
        }
    }
    Json::Value disable = root["layer"]["disable_environment"];
    if (!disable.isNull()) {
        for (Json::Value::iterator dis_iter = disable.begin(); dis_iter != disable.end(); dis_iter++) {
            if (dis_iter.key().isNull()) {
                continue;
            }
            disable_env_variable = dis_iter.key().asString();
#ifdef _WIN32
            if (0 != GetEnvironmentVariableA(disable_env_variable.c_str(), disable_env_value, 15)) {
#else
            char *disable_env = getenv(disable_env_variable.c_str());
            if (NULL != disable_env) {
                strncpy(disable_env_value, disable_env, 15);
                disable_env_value[15] = '\0';
#endif
                if (atoi(disable_env_value) > 0) {
                    disable_var_set = true;
                    enabled = false;
                }
            }
            break;
        }
    }

    // Print the overall state (ENABLED or DISABLED) so we can
    // quickly determine if this layer is being used.
    PrintBeginTableRow();
    PrintTableElement("");
    PrintTableElement("Enabled State");
    PrintTableElement(enabled ? "ENABLED" : "DISABLED");
    PrintTableElement("");
    PrintEndTableRow();
    PrintBeginTableRow();
    PrintTableElement("");
    PrintTableElement("Enable Env Var", ALIGN_RIGHT);
    PrintTableElement(enable_env_variable);
    if (enable_var_set) {
        PrintTableElement("");
    } else {
        PrintTableElement("Not Defined");
    }
    PrintEndTableRow();
    PrintBeginTableRow();
    PrintTableElement("");
    PrintTableElement("Disable Env Var", ALIGN_RIGHT);
    PrintTableElement(disable_env_variable);
    if (disable_var_set) {
        PrintTableElement(disable_env_value);
    } else {
        PrintTableElement("Not Defined");
    }
    PrintEndTableRow();
}

// Perform Vulkan commands to find out what extensions are available
// to a Vulkan Instance, and attempt to create one.
ErrorResults PrintInstanceInfo(void) {
    ErrorResults res = SUCCESSFUL;
    VkApplicationInfo app_info;
    VkInstanceCreateInfo inst_info;
    uint32_t ext_count;
    std::vector<VkExtensionProperties> ext_props;
    VkResult status;
    char generic_string[MAX_STRING_LENGTH];

    memset(&app_info, 0, sizeof(VkApplicationInfo));
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pNext = NULL;
    app_info.pApplicationName = "via";
    app_info.applicationVersion = 1;
    app_info.pEngineName = "via";
    app_info.engineVersion = 1;
    app_info.apiVersion = VK_API_VERSION_1_0;

    memset(&inst_info, 0, sizeof(VkInstanceCreateInfo));
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pNext = NULL;
    inst_info.pApplicationInfo = &app_info;
    inst_info.enabledLayerCount = 0;
    inst_info.ppEnabledLayerNames = NULL;
    inst_info.enabledExtensionCount = 0;
    inst_info.ppEnabledExtensionNames = NULL;

    PrintBeginTable("Instance", 3);

    PrintBeginTableRow();
    PrintTableElement("vkEnumerateInstanceExtensionProperties");
    status = vkEnumerateInstanceExtensionProperties(NULL, &ext_count, NULL);
    if (status) {
        snprintf(generic_string, MAX_STRING_LENGTH - 1, "ERROR: Failed to determine num inst extensions - %d", status);
        PrintTableElement(generic_string);
        PrintTableElement("");
        PrintEndTableRow();
        res = VULKAN_CANT_FIND_EXTENSIONS;
    } else {
        snprintf(generic_string, MAX_STRING_LENGTH - 1, "%d extensions found", ext_count);
        PrintTableElement(generic_string);
        PrintTableElement("");
        PrintEndTableRow();

        ext_props.resize(ext_count);
        status = vkEnumerateInstanceExtensionProperties(NULL, &ext_count, ext_props.data());
        if (status) {
            PrintBeginTableRow();
            PrintTableElement("");
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "ERROR: Failed to enumerate inst extensions - %d", status);
            PrintTableElement(generic_string);
            PrintTableElement("");
            PrintEndTableRow();
            res = VULKAN_CANT_FIND_EXTENSIONS;
        } else {
            for (uint32_t iii = 0; iii < ext_count; iii++) {
                PrintBeginTableRow();
                snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", iii);
                PrintTableElement(generic_string, ALIGN_RIGHT);
                PrintTableElement(ext_props[iii].extensionName);
                snprintf(generic_string, MAX_STRING_LENGTH - 1, "Spec Vers %d", ext_props[iii].specVersion);
                PrintTableElement(generic_string);
                PrintEndTableRow();
            }
        }
    }

    PrintBeginTableRow();
    PrintTableElement("vkCreateInstance");
    status = vkCreateInstance(&inst_info, NULL, &global_items.instance);
    if (status == VK_ERROR_INCOMPATIBLE_DRIVER) {
        PrintTableElement("ERROR: Incompatible Driver");
        res = VULKAN_CANT_FIND_DRIVER;
    } else if (status == VK_ERROR_OUT_OF_HOST_MEMORY) {
        PrintTableElement("ERROR: Out of memory");
        res = VULKAN_FAILED_OUT_OF_MEM;
    } else if (status) {
        snprintf(generic_string, MAX_STRING_LENGTH - 1, "ERROR: Failed to create - %d", status);
        PrintTableElement(generic_string);
        res = VULKAN_FAILED_CREATE_INSTANCE;
    } else {
        PrintTableElement("SUCCESSFUL");
    }
    PrintTableElement("");
    PrintEndTableRow();
    PrintEndTable();

    return res;
}

// Print out any information we can find out about physical devices using
// the Vulkan commands.  There should be one for each Vulkan capable device
// on the system.
ErrorResults PrintPhysDevInfo(void) {
    ErrorResults res = SUCCESSFUL;
    VkPhysicalDeviceProperties props;
    std::vector<VkPhysicalDevice> phys_devices;
    VkResult status;
    char generic_string[MAX_STRING_LENGTH];
    uint32_t gpu_count = 0;
    uint32_t iii;
    uint32_t jjj;

    PrintBeginTable("Physical Devices", 4);

    PrintBeginTableRow();
    PrintTableElement("vkEnumeratePhysicalDevices");
    status = vkEnumeratePhysicalDevices(global_items.instance, &gpu_count, NULL);
    if (status) {
        snprintf(generic_string, MAX_STRING_LENGTH - 1, "ERROR: Failed to query - %d", status);
        PrintTableElement(generic_string);
        res = VULKAN_CANT_FIND_DRIVER;
        goto out;
    } else {
        snprintf(generic_string, MAX_STRING_LENGTH - 1, "%d", gpu_count);
        PrintTableElement(generic_string);
    }
    PrintTableElement("");
    PrintTableElement("");
    PrintEndTableRow();

    phys_devices.resize(gpu_count);
    global_items.phys_devices.resize(gpu_count);
    status = vkEnumeratePhysicalDevices(global_items.instance, &gpu_count, phys_devices.data());
    if (VK_SUCCESS != status && VK_INCOMPLETE != status) {
        PrintBeginTableRow();
        PrintTableElement("");
        PrintTableElement("Failed to enumerate physical devices!");
        PrintTableElement("");
        PrintEndTableRow();
        res = VULKAN_CANT_FIND_DRIVER;
        goto out;
    }
    for (iii = 0; iii < gpu_count; iii++) {
        global_items.phys_devices[iii].vulkan_phys_dev = phys_devices[iii];

        PrintBeginTableRow();
        snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", iii);
        PrintTableElement(generic_string, ALIGN_RIGHT);
        if (status) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "ERROR: Failed to query - %d", status);
            PrintTableElement(generic_string);
            PrintTableElement("");
            PrintTableElement("");
            PrintEndTableRow();
        } else {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "0x%p", phys_devices[iii]);
            PrintTableElement(generic_string);
            PrintTableElement("");
            PrintTableElement("");
            PrintEndTableRow();

            vkGetPhysicalDeviceProperties(phys_devices[iii], &props);

            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Vendor");
            switch (props.vendorID) {
                case 0x8086:
                case 0x8087:
                    snprintf(generic_string, MAX_STRING_LENGTH - 1, "Intel [0x%04x]", props.vendorID);
                    break;
                case 0x1002:
                case 0x1022:
                    snprintf(generic_string, MAX_STRING_LENGTH - 1, "AMD [0x%04x]", props.vendorID);
                    break;
                case 0x10DE:
                    snprintf(generic_string, MAX_STRING_LENGTH - 1, "Nvidia [0x%04x]", props.vendorID);
                    break;
                case 0x1EB5:
                    snprintf(generic_string, MAX_STRING_LENGTH - 1, "ARM [0x%04x]", props.vendorID);
                    break;
                case 0x5143:
                    snprintf(generic_string, MAX_STRING_LENGTH - 1, "Qualcomm [0x%04x]", props.vendorID);
                    break;
                case 0x1099:
                case 0x10C3:
                case 0x1249:
                case 0x4E8:
                    snprintf(generic_string, MAX_STRING_LENGTH - 1, "Samsung [0x%04x]", props.vendorID);
                    break;
                default:
                    snprintf(generic_string, MAX_STRING_LENGTH - 1, "0x%04x", props.vendorID);
                    break;
            }
            PrintTableElement(generic_string);
            PrintTableElement("");
            PrintEndTableRow();

            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Device Name");
            PrintTableElement(props.deviceName);
            PrintTableElement("");
            PrintEndTableRow();

            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Device ID");
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "0x%x", props.deviceID);
            PrintTableElement(generic_string);
            PrintTableElement("");
            PrintEndTableRow();

            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Device Type");
            switch (props.deviceType) {
                case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                    PrintTableElement("Integrated GPU");
                    break;
                case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                    PrintTableElement("Discrete GPU");
                    break;
                case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                    PrintTableElement("Virtual GPU");
                    break;
                case VK_PHYSICAL_DEVICE_TYPE_CPU:
                    PrintTableElement("CPU");
                    break;
                case VK_PHYSICAL_DEVICE_TYPE_OTHER:
                    PrintTableElement("Other");
                    break;
                default:
                    PrintTableElement("INVALID!");
                    break;
            }
            PrintTableElement("");
            PrintEndTableRow();

            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Driver Version");
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%d.%d.%d", VK_VERSION_MAJOR(props.driverVersion),
                     VK_VERSION_MINOR(props.driverVersion), VK_VERSION_PATCH(props.driverVersion));
            PrintTableElement(generic_string);
            PrintTableElement("");
            PrintEndTableRow();

            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("API Version");
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%d.%d.%d", VK_VERSION_MAJOR(props.apiVersion),
                     VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion));
            PrintTableElement(generic_string);
            PrintTableElement("");
            PrintEndTableRow();

            uint32_t queue_fam_count;
            vkGetPhysicalDeviceQueueFamilyProperties(phys_devices[iii], &queue_fam_count, NULL);
            if (queue_fam_count > 0) {
                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement("Queue Families");
                snprintf(generic_string, MAX_STRING_LENGTH - 1, "%d", queue_fam_count);
                PrintTableElement(generic_string);
                PrintTableElement("");
                PrintEndTableRow();

                global_items.phys_devices[iii].queue_fam_props.resize(queue_fam_count);
                vkGetPhysicalDeviceQueueFamilyProperties(phys_devices[iii], &queue_fam_count,
                                                         global_items.phys_devices[iii].queue_fam_props.data());
                for (jjj = 0; jjj < queue_fam_count; jjj++) {
                    PrintBeginTableRow();
                    PrintTableElement("");
                    snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", jjj);
                    PrintTableElement(generic_string, ALIGN_RIGHT);
                    PrintTableElement("Queue Count");
                    snprintf(generic_string, MAX_STRING_LENGTH - 1, "%d",
                             global_items.phys_devices[iii].queue_fam_props[jjj].queueCount);
                    PrintTableElement(generic_string);
                    PrintEndTableRow();

                    PrintBeginTableRow();
                    PrintTableElement("");
                    PrintTableElement("");
                    PrintTableElement("Queue Flags");
                    generic_string[0] = '\0';
                    bool prev_set = false;
                    if (global_items.phys_devices[iii].queue_fam_props[jjj].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                        strncat(generic_string, "GRAPHICS", MAX_STRING_LENGTH - 1);
                        prev_set = true;
                    }
                    if (global_items.phys_devices[iii].queue_fam_props[jjj].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                        if (prev_set) {
                            strncat(generic_string, " | ", MAX_STRING_LENGTH - 1);
                        }
                        strncat(generic_string, "COMPUTE", MAX_STRING_LENGTH - 1);
                        prev_set = true;
                    }
                    if (global_items.phys_devices[iii].queue_fam_props[jjj].queueFlags & VK_QUEUE_TRANSFER_BIT) {
                        if (prev_set) {
                            strncat(generic_string, " | ", MAX_STRING_LENGTH - 1);
                        }
                        strncat(generic_string, "TRANSFER", MAX_STRING_LENGTH - 1);
                        prev_set = true;
                    }
                    if (global_items.phys_devices[iii].queue_fam_props[jjj].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) {
                        if (prev_set) {
                            strncat(generic_string, " | ", MAX_STRING_LENGTH - 1);
                        }
                        strncat(generic_string, "SPARSE_BINDING", MAX_STRING_LENGTH - 1);
                        prev_set = true;
                    }
                    if (!prev_set) {
                        strncat(generic_string, "--NONE--", MAX_STRING_LENGTH - 1);
                    }
                    PrintTableElement(generic_string);
                    PrintEndTableRow();

                    PrintBeginTableRow();
                    PrintTableElement("");
                    PrintTableElement("");
                    PrintTableElement("Timestamp Valid Bits");
                    snprintf(generic_string, MAX_STRING_LENGTH - 1, "0x%x",
                             global_items.phys_devices[iii].queue_fam_props[jjj].timestampValidBits);
                    PrintTableElement(generic_string);
                    PrintEndTableRow();

                    PrintBeginTableRow();
                    PrintTableElement("");
                    PrintTableElement("");
                    PrintTableElement("Image Granularity");
                    PrintTableElement("");
                    PrintEndTableRow();

                    PrintBeginTableRow();
                    PrintTableElement("");
                    PrintTableElement("");
                    PrintTableElement("Width", ALIGN_RIGHT);
                    snprintf(generic_string, MAX_STRING_LENGTH - 1, "0x%x",
                             global_items.phys_devices[iii].queue_fam_props[jjj].minImageTransferGranularity.width);
                    PrintTableElement(generic_string);
                    PrintEndTableRow();

                    PrintBeginTableRow();
                    PrintTableElement("");
                    PrintTableElement("");
                    PrintTableElement("Height", ALIGN_RIGHT);
                    snprintf(generic_string, MAX_STRING_LENGTH - 1, "0x%x",
                             global_items.phys_devices[iii].queue_fam_props[jjj].minImageTransferGranularity.height);
                    PrintTableElement(generic_string);
                    PrintEndTableRow();

                    PrintBeginTableRow();
                    PrintTableElement("");
                    PrintTableElement("");
                    PrintTableElement("Depth", ALIGN_RIGHT);
                    snprintf(generic_string, MAX_STRING_LENGTH - 1, "0x%x",
                             global_items.phys_devices[iii].queue_fam_props[jjj].minImageTransferGranularity.depth);
                    PrintTableElement(generic_string);
                    PrintEndTableRow();
                }
            } else {
                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement("vkGetPhysicalDeviceQueueFamilyProperties");
                PrintTableElement("FAILED: Returned 0!");
                PrintTableElement("");
                PrintEndTableRow();
            }

            VkPhysicalDeviceMemoryProperties memory_props;
            vkGetPhysicalDeviceMemoryProperties(phys_devices[iii], &memory_props);

            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Memory Heaps");
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%d", memory_props.memoryHeapCount);
            PrintTableElement(generic_string);
            PrintTableElement("");
            PrintEndTableRow();

            for (jjj = 0; jjj < memory_props.memoryHeapCount; jjj++) {
                PrintBeginTableRow();
                PrintTableElement("");
                snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", jjj);
                PrintTableElement(generic_string, ALIGN_RIGHT);
                PrintTableElement("Property Flags");
                generic_string[0] = '\0';
                bool prev_set = false;
                if (memory_props.memoryHeaps[jjj].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                    strncat(generic_string, "DEVICE_LOCAL", MAX_STRING_LENGTH - 1);
                    prev_set = true;
                }
                if (!prev_set) {
                    strncat(generic_string, "--NONE--", MAX_STRING_LENGTH - 1);
                }
                PrintTableElement(generic_string);
                PrintEndTableRow();

                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement("");
                PrintTableElement("Heap Size");
                snprintf(generic_string, MAX_STRING_LENGTH - 1, "%" PRIu64 "",
                         static_cast<uint64_t>(memory_props.memoryHeaps[jjj].size));
                PrintTableElement(generic_string);
                PrintEndTableRow();
            }

            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Memory Types");
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "%d", memory_props.memoryTypeCount);
            PrintTableElement(generic_string);
            PrintTableElement("");
            PrintEndTableRow();

            for (jjj = 0; jjj < memory_props.memoryTypeCount; jjj++) {
                PrintBeginTableRow();
                PrintTableElement("");
                snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", jjj);
                PrintTableElement(generic_string, ALIGN_RIGHT);
                PrintTableElement("Property Flags");
                generic_string[0] = '\0';
                bool prev_set = false;
                if (memory_props.memoryTypes[jjj].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                    strncat(generic_string, "DEVICE_LOCAL", MAX_STRING_LENGTH - 1);
                    prev_set = true;
                }
                if (memory_props.memoryTypes[jjj].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
                    if (prev_set) {
                        strncat(generic_string, " | ", MAX_STRING_LENGTH - 1);
                    }
                    strncat(generic_string, "HOST_VISIBLE", MAX_STRING_LENGTH - 1);
                    prev_set = true;
                }
                if (memory_props.memoryTypes[jjj].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
                    if (prev_set) {
                        strncat(generic_string, " | ", MAX_STRING_LENGTH - 1);
                    }
                    strncat(generic_string, "HOST_COHERENT", MAX_STRING_LENGTH - 1);
                    prev_set = true;
                }
                if (memory_props.memoryTypes[jjj].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) {
                    if (prev_set) {
                        strncat(generic_string, " | ", MAX_STRING_LENGTH - 1);
                    }
                    strncat(generic_string, "HOST_CACHED", MAX_STRING_LENGTH - 1);
                    prev_set = true;
                }
                if (memory_props.memoryTypes[jjj].propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) {
                    if (prev_set) {
                        strncat(generic_string, " | ", MAX_STRING_LENGTH - 1);
                    }
                    strncat(generic_string, "LAZILY_ALLOC", MAX_STRING_LENGTH - 1);
                    prev_set = true;
                }
                if (!prev_set) {
                    strncat(generic_string, "--NONE--", MAX_STRING_LENGTH - 1);
                }
                PrintTableElement(generic_string);
                PrintEndTableRow();

                PrintBeginTableRow();
                PrintTableElement("");
                PrintTableElement("");
                PrintTableElement("Heap Index");
                snprintf(generic_string, MAX_STRING_LENGTH - 1, "%d", memory_props.memoryTypes[jjj].heapIndex);
                PrintTableElement(generic_string);
                PrintEndTableRow();
            }

            uint32_t num_ext_props;
            std::vector<VkExtensionProperties> ext_props;

            PrintBeginTableRow();
            PrintTableElement("");
            PrintTableElement("Device Extensions");
            status = vkEnumerateDeviceExtensionProperties(phys_devices[iii], NULL, &num_ext_props, NULL);
            if (VK_SUCCESS != status) {
                PrintTableElement("FAILED querying number of extensions");
                PrintTableElement("");
                PrintEndTableRow();

                res = VULKAN_CANT_FIND_EXTENSIONS;
            } else {
                snprintf(generic_string, MAX_STRING_LENGTH - 1, "%d", num_ext_props);
                PrintTableElement(generic_string);
                ext_props.resize(num_ext_props);
                status = vkEnumerateDeviceExtensionProperties(phys_devices[iii], NULL, &num_ext_props, ext_props.data());
                if (VK_SUCCESS != status) {
                    PrintTableElement("FAILED querying actual extension info");
                    PrintEndTableRow();

                    res = VULKAN_CANT_FIND_EXTENSIONS;
                } else {
                    PrintTableElement("");
                    PrintEndTableRow();

                    for (jjj = 0; jjj < num_ext_props; jjj++) {
                        PrintBeginTableRow();
                        PrintTableElement("");
                        snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", jjj);
                        PrintTableElement(generic_string, ALIGN_RIGHT);
                        PrintTableElement(ext_props[jjj].extensionName);
                        snprintf(generic_string, MAX_STRING_LENGTH - 1, "Spec Vers %d", ext_props[jjj].specVersion);
                        PrintTableElement(generic_string);
                        PrintEndTableRow();
                    }
                }
            }
        }
    }

    PrintEndTable();

out:

    return res;
}

// Using the previously determine information, attempt to create a logical
// device for each physical device we found.
ErrorResults PrintLogicalDeviceInfo(void) {
    ErrorResults res = SUCCESSFUL;
    VkDeviceCreateInfo device_create_info;
    VkDeviceQueueCreateInfo queue_create_info;
    VkResult status = VK_SUCCESS;
    uint32_t dev_count = static_cast<uint32_t>(global_items.phys_devices.size());
    char generic_string[MAX_STRING_LENGTH];
    bool found_driver = false;

    PrintBeginTable("Logical Devices", 3);

    PrintBeginTableRow();
    PrintTableElement("vkCreateDevice");
    snprintf(generic_string, MAX_STRING_LENGTH - 1, "%d", dev_count);
    PrintTableElement(generic_string);
    PrintTableElement("");
    PrintEndTableRow();

    global_items.log_devices.resize(dev_count);
    for (uint32_t dev = 0; dev < dev_count; dev++) {
        memset(&device_create_info, 0, sizeof(VkDeviceCreateInfo));
        device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_create_info.pNext = NULL;
        device_create_info.queueCreateInfoCount = 0;
        device_create_info.pQueueCreateInfos = NULL;
        device_create_info.enabledLayerCount = 0;
        device_create_info.ppEnabledLayerNames = NULL;
        device_create_info.enabledExtensionCount = 0;
        device_create_info.ppEnabledExtensionNames = NULL;
        device_create_info.queueCreateInfoCount = 1;
        device_create_info.enabledLayerCount = 0;
        device_create_info.ppEnabledLayerNames = NULL;
        device_create_info.enabledExtensionCount = 0;
        device_create_info.ppEnabledExtensionNames = NULL;

        memset(&queue_create_info, 0, sizeof(VkDeviceQueueCreateInfo));
        float queue_priority = 0;
        queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info.pNext = NULL;
        queue_create_info.queueCount = 1;
        queue_create_info.pQueuePriorities = &queue_priority;

        for (uint32_t queue = 0; queue < global_items.phys_devices[dev].queue_fam_props.size(); queue++) {
            if (0 != (global_items.phys_devices[dev].queue_fam_props[queue].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                queue_create_info.queueFamilyIndex = queue;
                break;
            }
        }
        device_create_info.pQueueCreateInfos = &queue_create_info;

        PrintBeginTableRow();
        PrintTableElement("");
        snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", dev);
        PrintTableElement(generic_string);

        status = vkCreateDevice(global_items.phys_devices[dev].vulkan_phys_dev, &device_create_info, NULL,
                                &global_items.log_devices[dev]);
        if (VK_ERROR_INCOMPATIBLE_DRIVER == status) {
            PrintTableElement("FAILED: Incompatible Driver");
            if (!found_driver) {
                res = VULKAN_CANT_FIND_DRIVER;
            }
        } else if (VK_ERROR_OUT_OF_HOST_MEMORY == status) {
            PrintTableElement("FAILED: Out of Host Memory");
            // If we haven't already found a driver, set an error
            if (!found_driver) {
                res = VULKAN_FAILED_OUT_OF_MEM;
            }
        } else if (VK_SUCCESS != status) {
            snprintf(generic_string, MAX_STRING_LENGTH - 1, "FAILED : VkResult code = 0x%x", status);
            PrintTableElement(generic_string);
            // If we haven't already found a driver, set an error
            if (!found_driver) {
                res = VULKAN_FAILED_CREATE_DEVICE;
            }
        } else {
            PrintTableElement("SUCCESSFUL");
            found_driver = true;
            // Clear any potential previous errors
            res = SUCCESSFUL;
        }

        PrintEndTableRow();
    }

    PrintEndTable();

    return res;
}

// Clean up all the Vulkan items we previously created and print
// out if there are any problems.
void PrintCleanupInfo(void) {
    char generic_string[MAX_STRING_LENGTH];
    uint32_t dev_count = static_cast<uint32_t>(global_items.phys_devices.size());

    PrintBeginTable("Cleanup", 3);

    PrintBeginTableRow();
    PrintTableElement("vkDestroyDevice");
    snprintf(generic_string, MAX_STRING_LENGTH - 1, "%d", dev_count);
    PrintTableElement(generic_string);
    PrintTableElement("");
    PrintEndTableRow();
    for (uint32_t dev = 0; dev < dev_count; dev++) {
        vkDestroyDevice(global_items.log_devices[dev], NULL);
        PrintBeginTableRow();
        PrintTableElement("");
        snprintf(generic_string, MAX_STRING_LENGTH - 1, "[%d]", dev);
        PrintTableElement(generic_string, ALIGN_RIGHT);
        PrintTableElement("SUCCESSFUL");
        PrintEndTableRow();
    }

    PrintBeginTableRow();
    PrintTableElement("vkDestroyInstance");
    vkDestroyInstance(global_items.instance, NULL);
    PrintTableElement("SUCCESSFUL");
    PrintTableElement("");
    PrintEndTableRow();

    PrintEndTable();
}

// Run any external tests we can find, and print the results of those
// tests.
ErrorResults PrintTestResults(void) {
    ErrorResults res = SUCCESSFUL;

    BeginSection("External Tests");
    if (global_items.sdk_found) {
        std::string cube_exe;
        std::string full_cmd;
        std::string path = global_items.sdk_path;

#ifdef _WIN32
        cube_exe = "cube.exe";

#if _WIN64
        path += "\\Bin";
#else
        path += "\\Bin32";
#endif
#else  // gcc
        cube_exe = "./cube";
        path += "/../examples/build";
#endif
        full_cmd = cube_exe;
        full_cmd += " --c 100";

        PrintBeginTable("Cube", 2);

        PrintBeginTableRow();
        PrintTableElement(full_cmd);
        int test_result = RunTestInDirectory(path, cube_exe, full_cmd);
        if (test_result == 0) {
            PrintTableElement("SUCCESSFUL");
        } else if (test_result == 1) {
            PrintTableElement("Not Found");
        } else {
            PrintTableElement("FAILED!");
            res = TEST_FAILED;
        }
        PrintEndTableRow();

        full_cmd += " --validate";

        PrintBeginTableRow();
        PrintTableElement(full_cmd);
        test_result = RunTestInDirectory(path, cube_exe, full_cmd);
        if (test_result == 0) {
            PrintTableElement("SUCCESSFUL");
        } else if (test_result == 1) {
            PrintTableElement("Not Found");
        } else {
            PrintTableElement("FAILED!");
            res = TEST_FAILED;
        }
        PrintEndTableRow();

        PrintEndTable();
    } else {
        PrintStandardText("No SDK found by VIA, skipping test section");
    }
    EndSection();

    return res;
}

// Print information on any Vulkan commands we can (or can't) execute.
ErrorResults PrintVulkanInfo(void) {
    ErrorResults res = SUCCESSFUL;
    bool created = false;
    BeginSection("Vulkan API Calls");

    res = PrintInstanceInfo();
    if (res != SUCCESSFUL) {
        goto out;
    }
    created = true;
    res = PrintPhysDevInfo();
    if (res != SUCCESSFUL) {
        goto out;
    }
    res = PrintLogicalDeviceInfo();
    if (res != SUCCESSFUL) {
        goto out;
    }

out:
    if (created) {
        PrintCleanupInfo();
    }

    EndSection();

    return res;
}
