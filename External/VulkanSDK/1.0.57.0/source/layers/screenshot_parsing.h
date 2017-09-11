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

#include <string.h>
#include <assert.h>
#include <iostream>

namespace screenshot {

static const int SCREEN_SHOT_FRAMES_INTERVAL_DEFAULT = 1;
static const int SCREEN_SHOT_FRAMES_UNLIMITED = -1;

typedef struct {
    bool valid;
    int startFrame;  // the range begin from (include) this frame.
    int count;       // if the value is SCREEN_SHOT_FRAMES_UNLIMITED, it means unlimited screenshots until capture/playback to end.
    int interval;
} FrameRange;

// initialize pFrameRange, parse rangeString and set value to members of *pFrameRange.
// the string of rangeString can be and must be one of the following values:
// 1. all
// 2. <startFrame>-<frameCount>-<interval>
// 3. <startFrame>-<frameCount>
//    if frameCount is 0, it means the range is unlimited range or all frames from startFrame.
// return:
// return 0 if parsing rangeString successfully, other value is a status value indicating a specified error was encountered,
// currently support the following values:
//        1, parsing error or input parameters less than two.
//        2, start frame number < 0.
//        3, frameCount < 0.
//        4, interval <= 0
//        .......
int initScreenShotFrameRange(const char *rangeString, FrameRange *pFrameRange);

// detect if the input command line option _vk_screenshot is definition of frame range or a frame list.
bool isOptionBelongToScreenShotRange(const char *_vk_screenshot);

// check screenshot frame range command line option
// return:
//      indicate check success or not. if fail, return false.
bool checkParsingFrameRange(const char *_vk_screenshot);
}
