/*
* Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.
* Copyright (C) 2015-2016 LunarG, Inc.
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

#include "screenshot_parsing.h"

using namespace std;

namespace screenshot {

// check rangeString:
//    1> rangeString is valid and every char is '0'-'9' or '-'
//    2> char '-' count <= 2;
// return:
//    false if 1> or 2> fail;
bool checkScreenShotFrameRangeString(const char *rangeString) {
    bool checkResult = false;
    char *currentChar = const_cast<char *>(rangeString);
    int dashCount = 0;

    if (rangeString != nullptr) {
        checkResult = true;
        while (*currentChar != '\0') {
            if ((*currentChar >= '0') && (*currentChar <= '9')) {
                currentChar++;
            } else if (*currentChar == '-') {
                dashCount++;
                currentChar++;
            } else {
                checkResult = false;
                break;
            }
        }
        checkResult = checkResult && (dashCount <= 2);
    }
    return checkResult;
}

// initialize pFrameRange, parse rangeString and set value to members of *pFrameRange.
// the string of rangeString can be and must be one of the following values:
// 1. all
// 2. <startFrame>-<frameCount>-<interval>
//    if frameCount is 0, it means the range is unlimited range or all frames from startFrame.
// return:
// return 0 if parsing rangeString successfully, other value is a status value indicating a specified error was encountered,
// currently support the following values:
//        1, parsing error or input parameters include negative value or parameter number is not two or three.
//        2, start frame number < 0.
//        3, frameCount < 0.
//        4, interval <= 0
//        .......
int initScreenShotFrameRange(const char *rangeString, FrameRange *pFrameRange) {
    int parsingStatus = 0;
    if (rangeString && *rangeString) {
        string parameter(rangeString);
        pFrameRange->valid = false;
        if ((!parameter.empty())) {
            if (parameter.compare("all") == 0) {
                pFrameRange->valid = true;
                pFrameRange->startFrame = 0;
                pFrameRange->count = SCREEN_SHOT_FRAMES_UNLIMITED;
                pFrameRange->interval = SCREEN_SHOT_FRAMES_INTERVAL_DEFAULT;
            } else {
                if (!checkScreenShotFrameRangeString(rangeString)) {
                    parsingStatus = 1;
                } else {
                    int frameCount = 0;
                    int itemCount =
                        sscanf(parameter.c_str(), "%d-%d-%d", &pFrameRange->startFrame, &frameCount, &pFrameRange->interval);
                    if (itemCount >= 2) {
                        if (itemCount == 2) {
                            pFrameRange->interval = SCREEN_SHOT_FRAMES_INTERVAL_DEFAULT;
                        }

                        if ((pFrameRange->startFrame < 0) || (frameCount < 0) || (pFrameRange->interval <= 0)) {
                            if (pFrameRange->startFrame < 0) {
                                parsingStatus = 2;
                            } else if (frameCount < 0) {
                                parsingStatus = 3;
                            } else {
                                parsingStatus = 4;
                            }
                        } else {
                            pFrameRange->valid = true;
                        }

                        if (parsingStatus == 0) {
                            if (frameCount == 0) {
                                pFrameRange->count = SCREEN_SHOT_FRAMES_UNLIMITED;
                            } else {
                                pFrameRange->count = frameCount / pFrameRange->interval;
                                if ((frameCount % pFrameRange->interval) != 0) {
                                    pFrameRange->count++;
                                }
                            }
                        }
                    } else {
                        parsingStatus = 1;
                    }
                }
            }
        }
    }
    return parsingStatus;
}

// detect if the input command option _vk_screenshot is definition of frame range or a frame list.
bool isOptionBelongToScreenShotRange(const char *_vk_screenshot) {
    bool belongToScreenShotRange = false;
    if ((strstr(_vk_screenshot, "-") != nullptr) || (strcmp(_vk_screenshot, "all") == 0)) {
        belongToScreenShotRange = true;
    }
    return belongToScreenShotRange;
}

// check screenshot frame range command line option
bool checkParsingFrameRange(const char *_vk_screenshot) {
    bool checkPassed = true;
    if (isOptionBelongToScreenShotRange(_vk_screenshot)) {
        screenshot::FrameRange frameRange;
        int parsingStatus = initScreenShotFrameRange(_vk_screenshot, &frameRange);
        if (parsingStatus != 0) {
            checkPassed = false;
        }
    }
    return checkPassed;
}
}
