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

#include <cassert>
#include <cmath>
#include <array>
#include <glm/gtc/matrix_transform.hpp>
#include "Simulation.h"

namespace {

class MeshPicker {
   public:
    MeshPicker()
        : pattern_({
              Meshes::MESH_PYRAMID, Meshes::MESH_ICOSPHERE, Meshes::MESH_TEAPOT, Meshes::MESH_PYRAMID, Meshes::MESH_ICOSPHERE,
              Meshes::MESH_PYRAMID, Meshes::MESH_PYRAMID, Meshes::MESH_PYRAMID, Meshes::MESH_PYRAMID, Meshes::MESH_PYRAMID,
          }),
          cur_(-1) {}

    Meshes::Type pick() {
        cur_ = (cur_ + 1) % pattern_.size();
        return pattern_[cur_];
    }

    float scale(Meshes::Type type) const {
        float base = 0.005f;

        switch (type) {
            case Meshes::MESH_PYRAMID:
            default:
                return base * 1.0f;
            case Meshes::MESH_ICOSPHERE:
                return base * 3.0f;
            case Meshes::MESH_TEAPOT:
                return base * 10.0f;
        }
    }

   private:
    const std::array<Meshes::Type, 10> pattern_;
    int cur_;
};

class ColorPicker {
   public:
    ColorPicker(unsigned int rng_seed) : rng_(rng_seed), red_(0.0f, 1.0f), green_(0.0f, 1.0f), blue_(0.0f, 1.0f) {}

    glm::vec3 pick() { return glm::vec3{red_(rng_), green_(rng_), blue_(rng_)}; }

   private:
    std::mt19937 rng_;
    std::uniform_real_distribution<float> red_;
    std::uniform_real_distribution<float> green_;
    std::uniform_real_distribution<float> blue_;
};

}  // namespace

Animation::Animation(unsigned int rng_seed, float scale) : rng_(rng_seed), dir_(-1.0f, 1.0f), speed_(0.1f, 1.0f) {
    float x = dir_(rng_);
    float y = dir_(rng_);
    float z = dir_(rng_);
    if (std::abs(x) + std::abs(y) + std::abs(z) == 0.0f) x = 1.0f;

    current_.axis = glm::normalize(glm::vec3(x, y, z));

    current_.speed = speed_(rng_);
    current_.scale = scale;

    current_.matrix = glm::scale(glm::mat4(1.0f), glm::vec3(current_.scale));

    current_.alpha = current_.speed;
    current_.alpha_inc = current_.alpha > 0.5f ? 0.05f : -0.05f;
}

float Animation::transparency() {
    if (current_.alpha <= 0.0f || current_.alpha >= 1.0f) {
        current_.alpha_inc *= -1.0f;
    }
    current_.alpha += current_.alpha_inc;
    return current_.alpha;
}

glm::mat4 Animation::transformation(float t) {
    current_.matrix = glm::rotate(current_.matrix, current_.speed * t, current_.axis);

    return current_.matrix;
}

class Curve {
   public:
    virtual ~Curve() {}
    virtual glm::vec3 evaluate(float t) = 0;
};

namespace {

enum CurveType {
    CURVE_RANDOM,
    CURVE_CIRCLE,
    CURVE_COUNT,
};

class RandomCurve : public Curve {
   public:
    RandomCurve(unsigned int rng_seed)
        : rng_(rng_seed),
          direction_(-0.3f, 0.3f),
          duration_(1.0f, 5.0f),
          segment_start_(0.0f),
          segment_direction_(0.0f),
          time_start_(0.0f),
          time_duration_(0.0f) {}

    glm::vec3 evaluate(float t) {
        if (t >= time_start_ + time_duration_) new_segment(t);

        pos_ += unit_dir_ * (t - last_);
        last_ = t;

        return pos_;
    }

   private:
    void new_segment(float time_start) {
        segment_start_ += segment_direction_;
        segment_direction_ = glm::vec3(direction_(rng_), direction_(rng_), direction_(rng_));

        time_start_ = time_start;
        time_duration_ = duration_(rng_);

        unit_dir_ = segment_direction_ / time_duration_;
        pos_ = segment_start_;
        last_ = time_start_;
    }

    std::mt19937 rng_;
    std::uniform_real_distribution<float> direction_;
    std::uniform_real_distribution<float> duration_;

    glm::vec3 segment_start_;
    glm::vec3 segment_direction_;
    float time_start_;
    float time_duration_;

    glm::vec3 unit_dir_;
    glm::vec3 pos_;
    float last_;
};

class CircleCurve : public Curve {
   public:
    CircleCurve(float radius, glm::vec3 axis) : r_(radius) {
        glm::vec3 a;

        if (axis.x != 0.0f) {
            a.x = -axis.z / axis.x;
            a.y = 0.0f;
            a.z = 1.0f;
        } else if (axis.y != 0.0f) {
            a.x = 1.0f;
            a.y = -axis.x / axis.y;
            a.z = 0.0f;
        } else {
            a.x = 1.0f;
            a.y = 0.0f;
            a.z = -axis.x / axis.z;
        }

        a_ = glm::normalize(a);
        b_ = glm::normalize(glm::cross(a_, axis));
    }

    glm::vec3 evaluate(float t) {
        return (a_ * (glm::vec3(std::cos(t)) - glm::vec3(1.0f)) + b_ * glm::vec3(std::sin(t))) * glm::vec3(r_);
    }

   private:
    float r_;
    glm::vec3 a_;
    glm::vec3 b_;
};

}  // namespace

Path::Path(unsigned int rng_seed) : rng_(rng_seed), type_(0, CURVE_COUNT - 1), duration_(5.0f, 20.0f) {
    // trigger a subpath generation
    current_.end = -1.0f;
    current_.now = 0.0f;
}

glm::vec3 Path::position(float t) {
    current_.now += t;

    while (current_.now >= current_.end) generate_subpath();

    return current_.origin + current_.curve->evaluate(current_.now - current_.start);
}

void Path::generate_subpath() {
    float duration = duration_(rng_);
    CurveType type = static_cast<CurveType>(type_(rng_));

    if (current_.curve) {
        current_.origin += current_.curve->evaluate(current_.end - current_.start);
        current_.start = current_.end;
    } else {
        std::uniform_real_distribution<float> origin(0.0f, 2.0f);
        current_.origin = glm::vec3(origin(rng_), origin(rng_), origin(rng_));
        current_.start = current_.now;
    }

    current_.end = current_.start + duration;

    Curve *curve;

    switch (type) {
        case CURVE_RANDOM:
            curve = new RandomCurve(rng_());
            break;
        case CURVE_CIRCLE: {
            std::uniform_real_distribution<float> dir(-1.0f, 1.0f);
            glm::vec3 axis(dir(rng_), dir(rng_), dir(rng_));
            if (axis.x == 0.0f && axis.y == 0.0f && axis.z == 0.0f) axis.x = 1.0f;

            std::uniform_real_distribution<float> radius_(0.02f, 0.2f);
            curve = new CircleCurve(radius_(rng_), axis);
        } break;
        default:
            assert(!"unreachable");
            curve = nullptr;
            break;
    }

    current_.curve.reset(curve);
}

Simulation::Simulation(int object_count) : random_dev_() {
    MeshPicker mesh;
    ColorPicker color(random_dev_());

    objects_.reserve(object_count);
    for (int i = 0; i < object_count; i++) {
        Meshes::Type type = mesh.pick();
        float scale = mesh.scale(type);

        objects_.emplace_back(Object{
            type, glm::vec3(0.5f + 0.5f * (float)i / object_count), color.pick(), Animation(random_dev_(), scale),
            Path(random_dev_()),
        });
    }
}

void Simulation::set_frame_data_size(uint32_t size) {
    uint32_t offset = 0;
    for (auto &obj : objects_) {
        obj.frame_data_offset = offset;
        offset += size;
    }
}

void Simulation::update(float time, int begin, int end) {
    for (int i = begin; i < end; i++) {
        auto &obj = objects_[i];

        glm::vec3 pos = obj.path.position(time);
        glm::mat4 trans = obj.animation.transformation(time);
        obj.model = glm::translate(glm::mat4(1.0f), pos) * trans;
        obj.alpha = obj.animation.transparency();
    }
}
