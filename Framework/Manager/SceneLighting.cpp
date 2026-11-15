#include "SceneLighting.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#include "Scene.hpp"

using namespace My;

namespace {
void BuildLightPerspective(Matrix4X4f& projection, float fieldOfView,
                           bool openGLProjection) {
    if (openGLProjection) {
        BuildOpenglPerspectiveFovRHMatrix(projection, fieldOfView, 1.0f, 1.0f,
                                          POINT_SHADOW_FAR_PLANE);
    } else {
        BuildPerspectiveFovRHMatrix(projection, fieldOfView, 1.0f, 1.0f,
                                    POINT_SHADOW_FAR_PLANE);
    }
}
}  // namespace

bool My::PrepareSceneLights(const Scene& scene, bool openGLProjection,
                            LightingFrame& lighting) {
    LightingFrame result;
    result.clipSpaceType = openGLProjection ? 0 : 1;
    for (const auto& entry : scene.LightNodes) {
        const auto node = entry.second.lock();
        if (!node) continue;
        if (result.numLights == MAX_LIGHTS) {
            std::cerr
                << "[SceneLighting] Scene exceeds light buffer capacity\n";
            return false;
        }
        const auto source = scene.GetLight(node->GetSceneObjectRef());
        if (!source) {
            std::cerr << "[SceneLighting] Unresolved light reference: "
                      << node->GetSceneObjectRef() << '\n';
            return false;
        }

        Light& light = result.lightInfo.lights[result.numLights];
        const auto transform = *node->GetCalculatedTransform();
        light.lightPosition = {0.0f, 0.0f, 0.0f, 1.0f};
        light.lightDirection = {0.0f, 0.0f, -1.0f, 0.0f};
        Transform(light.lightPosition, transform);
        Transform(light.lightDirection, transform);
        Normalize(light.lightDirection);

        light.lightGuid = source->GetGuid();
        light.lightColor = source->GetColor().Value;
        light.lightIntensity = source->GetIntensity();
        light.lightCastShadow = source->GetIfCastShadow();
        const auto& distance = source->GetDistanceAttenuation();
        light.lightDistAttenCurveType = distance.type;
        if (distance.type != AttenCurveType::kNone) {
            std::memcpy(light.lightDistAttenCurveParams, &distance.u,
                        sizeof(distance.u));
        }
        light.lightAngleAttenCurveType = AttenCurveType::kNone;
        BuildIdentityMatrix(light.lightProjectionMatrix);

        float nearClip = 1.0f;
        float farClip = POINT_SHADOW_FAR_PLANE;
        Vector4f target = light.lightPosition + light.lightDirection;
        Vector3f up = {0.0f, 0.0f, 1.0f};

        if (source->GetType() == SceneObjectType::kSceneObjectTypeLightInfi) {
            light.lightType = LightType::Infinity;
            target = {0.0f, 0.0f, 0.0f, 1.0f};
            const auto cameraNode = scene.GetFirstCameraNode();
            if (cameraNode) {
                const auto camera =
                    scene.GetCamera(cameraNode->GetSceneObjectRef());
                if (!camera) {
                    std::cerr << "[SceneLighting] Unresolved shadow camera\n";
                    return false;
                }
                nearClip = camera->GetNearClipDistance();
                farClip = camera->GetFarClipDistance();
                if (!std::isfinite(nearClip) || !std::isfinite(farClip) ||
                    nearClip <= 0.0f || farClip <= nearClip) {
                    std::cerr
                        << "[SceneLighting] Invalid shadow camera clips\n";
                    return false;
                }
                target[2] = -(0.75f * nearClip + 0.25f * farClip);
                Transform(target, *cameraNode->GetCalculatedTransform());
            }
            light.lightPosition = target - light.lightDirection * farClip;
            if (std::abs(light.lightDirection[0]) <= 0.2f &&
                std::abs(light.lightDirection[1]) <= 0.2f) {
                up = {0.1f, 0.1f, 1.0f};
            }
            const float halfDistance = (std::min)(farClip * 0.25f, 800.0f);
            if (openGLProjection) {
                BuildOpenglOrthographicRHMatrix(
                    light.lightProjectionMatrix, -halfDistance, halfDistance,
                    halfDistance, -halfDistance, nearClip,
                    farClip + halfDistance);
            } else {
                BuildOrthographicRHMatrix(light.lightProjectionMatrix,
                                          -halfDistance, halfDistance,
                                          halfDistance, -halfDistance, nearClip,
                                          farClip + halfDistance);
            }
        } else {
            if (std::abs(light.lightDirection[0]) <= 0.1f &&
                std::abs(light.lightDirection[1]) <= 0.1f) {
                up = {0.0f, 0.707f, 0.707f};
            }
            if (source->GetType() ==
                SceneObjectType::kSceneObjectTypeLightSpot) {
                const auto spot =
                    std::dynamic_pointer_cast<SceneObjectSpotLight>(source);
                if (!spot) {
                    std::cerr << "[SceneLighting] Invalid spot light object\n";
                    return false;
                }
                const auto& angle = spot->GetAngleAttenuation();
                if (angle.type != AttenCurveType::kNone) {
                    std::memcpy(light.lightAngleAttenCurveParams, &angle.u,
                                sizeof(angle.u));
                }
                light.lightAngleAttenCurveType = angle.type;
                const float fov = light.lightAngleAttenCurveParams[0][1] * 2.0f;
                if (!std::isfinite(fov) || fov <= 0.0f || fov >= PI) {
                    std::cerr << "[SceneLighting] Invalid spot light cone\n";
                    return false;
                }
                light.lightType = LightType::Spot;
                BuildLightPerspective(light.lightProjectionMatrix, fov,
                                      openGLProjection);
            } else if (source->GetType() ==
                       SceneObjectType::kSceneObjectTypeLightArea) {
                const auto area =
                    std::dynamic_pointer_cast<SceneObjectAreaLight>(source);
                if (!area) {
                    std::cerr << "[SceneLighting] Invalid area light object\n";
                    return false;
                }
                light.lightType = LightType::Area;
                light.lightSize = area->GetDimension();
            } else {
                light.lightType = LightType::Omni;
                BuildLightPerspective(light.lightProjectionMatrix, PI / 2.0f,
                                      openGLProjection);
            }
        }

        Vector3f position;
        position.Set(light.lightPosition);
        Vector3f lookAt;
        lookAt.Set(target);
        BuildViewRHMatrix(light.lightViewMatrix, position, lookAt, up);
        if (light.lightType == LightType::Infinity) {
            // The shader distinguishes a directional light using position W.
            light.lightPosition[3] = 0.0f;
        }
        ++result.numLights;
    }
    lighting = result;
    return true;
}
