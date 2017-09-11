/*
 * Vulkan Samples
 *
 * Copyright (C) 2015-2016 Valve Corporation
 * Copyright (C) 2015-2016 LunarG, Inc.
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

//--------------------------------------------------------------------------------------
// Mesh and VertexFormat Data
//--------------------------------------------------------------------------------------
struct Vertex {
    float posX, posY, posZ, posW;  // Position data
    float r, g, b, a;              // Color
};

struct VertexUV {
    float posX, posY, posZ, posW;  // Position data
    float u, v;                    // texture u,v
};

#define XYZ1(_x_, _y_, _z_) (_x_), (_y_), (_z_), 1.f
#define UV(_u_, _v_) (_u_), (_v_)

static const Vertex g_vbData[] = {
    {XYZ1(-1, -1, -1), XYZ1(0.f, 0.f, 0.f)}, {XYZ1(1, -1, -1), XYZ1(1.f, 0.f, 0.f)}, {XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f)},
    {XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f)},  {XYZ1(1, -1, -1), XYZ1(1.f, 0.f, 0.f)}, {XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f)},

    {XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f)},  {XYZ1(-1, 1, 1), XYZ1(0.f, 1.f, 1.f)},  {XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 1.f)},
    {XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 1.f)},   {XYZ1(-1, 1, 1), XYZ1(0.f, 1.f, 1.f)},  {XYZ1(1, 1, 1), XYZ1(1.f, 1.f, 1.f)},

    {XYZ1(1, 1, 1), XYZ1(1.f, 1.f, 1.f)},    {XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f)},  {XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 1.f)},
    {XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 1.f)},   {XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f)},  {XYZ1(1, -1, -1), XYZ1(1.f, 0.f, 0.f)},

    {XYZ1(-1, 1, 1), XYZ1(0.f, 1.f, 1.f)},   {XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f)}, {XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f)},
    {XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f)},  {XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f)}, {XYZ1(-1, -1, -1), XYZ1(0.f, 0.f, 0.f)},

    {XYZ1(1, 1, 1), XYZ1(1.f, 1.f, 1.f)},    {XYZ1(-1, 1, 1), XYZ1(0.f, 1.f, 1.f)},  {XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f)},
    {XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f)},   {XYZ1(-1, 1, 1), XYZ1(0.f, 1.f, 1.f)},  {XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f)},

    {XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 1.f)},   {XYZ1(1, -1, -1), XYZ1(1.f, 0.f, 0.f)}, {XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f)},
    {XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f)},  {XYZ1(1, -1, -1), XYZ1(1.f, 0.f, 0.f)}, {XYZ1(-1, -1, -1), XYZ1(0.f, 0.f, 0.f)},
};

static const Vertex g_vb_solid_face_colors_Data[] = {
    // red face
    {XYZ1(-1, -1, 1), XYZ1(1.f, 0.f, 0.f)},
    {XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 0.f)},
    {XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 0.f)},
    {XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 0.f)},
    {XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 0.f)},
    {XYZ1(1, 1, 1), XYZ1(1.f, 0.f, 0.f)},
    // green face
    {XYZ1(-1, -1, -1), XYZ1(0.f, 1.f, 0.f)},
    {XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 0.f)},
    {XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f)},
    {XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f)},
    {XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 0.f)},
    {XYZ1(1, 1, -1), XYZ1(0.f, 1.f, 0.f)},
    // blue face
    {XYZ1(-1, 1, 1), XYZ1(0.f, 0.f, 1.f)},
    {XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f)},
    {XYZ1(-1, 1, -1), XYZ1(0.f, 0.f, 1.f)},
    {XYZ1(-1, 1, -1), XYZ1(0.f, 0.f, 1.f)},
    {XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f)},
    {XYZ1(-1, -1, -1), XYZ1(0.f, 0.f, 1.f)},
    // yellow face
    {XYZ1(1, 1, 1), XYZ1(1.f, 1.f, 0.f)},
    {XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f)},
    {XYZ1(1, -1, 1), XYZ1(1.f, 1.f, 0.f)},
    {XYZ1(1, -1, 1), XYZ1(1.f, 1.f, 0.f)},
    {XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f)},
    {XYZ1(1, -1, -1), XYZ1(1.f, 1.f, 0.f)},
    // magenta face
    {XYZ1(1, 1, 1), XYZ1(1.f, 0.f, 1.f)},
    {XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 1.f)},
    {XYZ1(1, 1, -1), XYZ1(1.f, 0.f, 1.f)},
    {XYZ1(1, 1, -1), XYZ1(1.f, 0.f, 1.f)},
    {XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 1.f)},
    {XYZ1(-1, 1, -1), XYZ1(1.f, 0.f, 1.f)},
    // cyan face
    {XYZ1(1, -1, 1), XYZ1(0.f, 1.f, 1.f)},
    {XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 1.f)},
    {XYZ1(-1, -1, 1), XYZ1(0.f, 1.f, 1.f)},
    {XYZ1(-1, -1, 1), XYZ1(0.f, 1.f, 1.f)},
    {XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 1.f)},
    {XYZ1(-1, -1, -1), XYZ1(0.f, 1.f, 1.f)},
};

static const VertexUV g_vb_texture_Data[] = {
    // left face
    {XYZ1(-1, -1, -1), UV(1.f, 0.f)},  // lft-top-front
    {XYZ1(-1, 1, 1), UV(0.f, 1.f)},    // lft-btm-back
    {XYZ1(-1, -1, 1), UV(0.f, 0.f)},   // lft-top-back
    {XYZ1(-1, 1, 1), UV(0.f, 1.f)},    // lft-btm-back
    {XYZ1(-1, -1, -1), UV(1.f, 0.f)},  // lft-top-front
    {XYZ1(-1, 1, -1), UV(1.f, 1.f)},   // lft-btm-front
    // front face
    {XYZ1(-1, -1, -1), UV(0.f, 0.f)},  // lft-top-front
    {XYZ1(1, -1, -1), UV(1.f, 0.f)},   // rgt-top-front
    {XYZ1(1, 1, -1), UV(1.f, 1.f)},    // rgt-btm-front
    {XYZ1(-1, -1, -1), UV(0.f, 0.f)},  // lft-top-front
    {XYZ1(1, 1, -1), UV(1.f, 1.f)},    // rgt-btm-front
    {XYZ1(-1, 1, -1), UV(0.f, 1.f)},   // lft-btm-front
    // top face
    {XYZ1(-1, -1, -1), UV(0.f, 1.f)},  // lft-top-front
    {XYZ1(1, -1, 1), UV(1.f, 0.f)},    // rgt-top-back
    {XYZ1(1, -1, -1), UV(1.f, 1.f)},   // rgt-top-front
    {XYZ1(-1, -1, -1), UV(0.f, 1.f)},  // lft-top-front
    {XYZ1(-1, -1, 1), UV(0.f, 0.f)},   // lft-top-back
    {XYZ1(1, -1, 1), UV(1.f, 0.f)},    // rgt-top-back
    // bottom face
    {XYZ1(-1, 1, -1), UV(0.f, 0.f)},  // lft-btm-front
    {XYZ1(1, 1, 1), UV(1.f, 1.f)},    // rgt-btm-back
    {XYZ1(-1, 1, 1), UV(0.f, 1.f)},   // lft-btm-back
    {XYZ1(-1, 1, -1), UV(0.f, 0.f)},  // lft-btm-front
    {XYZ1(1, 1, -1), UV(1.f, 0.f)},   // rgt-btm-front
    {XYZ1(1, 1, 1), UV(1.f, 1.f)},    // rgt-btm-back
    // right face
    {XYZ1(1, 1, -1), UV(0.f, 1.f)},   // rgt-btm-front
    {XYZ1(1, -1, 1), UV(1.f, 0.f)},   // rgt-top-back
    {XYZ1(1, 1, 1), UV(1.f, 1.f)},    // rgt-btm-back
    {XYZ1(1, -1, 1), UV(1.f, 0.f)},   // rgt-top-back
    {XYZ1(1, 1, -1), UV(0.f, 1.f)},   // rgt-btm-front
    {XYZ1(1, -1, -1), UV(0.f, 0.f)},  // rgt-top-front
    // back face
    {XYZ1(-1, 1, 1), UV(1.f, 1.f)},   // lft-btm-back
    {XYZ1(1, 1, 1), UV(0.f, 1.f)},    // rgt-btm-back
    {XYZ1(-1, -1, 1), UV(1.f, 0.f)},  // lft-top-back
    {XYZ1(-1, -1, 1), UV(1.f, 0.f)},  // lft-top-back
    {XYZ1(1, 1, 1), UV(0.f, 1.f)},    // rgt-btm-back
    {XYZ1(1, -1, 1), UV(0.f, 0.f)},   // rgt-top-back
};
