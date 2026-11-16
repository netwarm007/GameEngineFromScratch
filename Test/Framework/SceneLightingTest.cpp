#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <string>

#include "GfxConfiguration.hpp"
#include "Scene.hpp"
#include "SceneLighting.hpp"

using namespace My;

namespace {
int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

bool Near(float actual, float expected) {
    return std::abs(actual - expected) < 0.0001f;
}

std::shared_ptr<SceneLightNode> AddLight(
    Scene& scene, const std::shared_ptr<SceneObjectLight>& source,
    const std::string& key = "light") {
    auto node = std::make_shared<SceneLightNode>("light node");
    node->AddSceneObjectRef(key);
    node->AppendTransform("position", std::make_shared<SceneObjectTranslation>(
                                          3.0f, -2.0f, 6.0f));
    scene.SceneGraph->AppendChild(node);
    scene.LightNodes.emplace(key, node);
    scene.Lights.emplace(key, source);
    return node;
}

void ExpectDepth(const Light& light, bool openGL, float nearClip,
                 float farClip) {
    Vector4f nearPoint{0, 0, -nearClip, 1};
    Vector4f farPoint{0, 0, -farClip, 1};
    Transform(nearPoint, light.lightProjectionMatrix);
    Transform(farPoint, light.lightProjectionMatrix);
    Expect(Near(nearPoint[2] / nearPoint[3], openGL ? -1.0f : 0.0f) &&
               Near(farPoint[2] / farPoint[3], 1.0f),
           "light projection maps near and far to backend depth endpoints");
}

void ExpectView(const Light& light, Vector4f target) {
    Vector4f position = light.lightPosition;
    position[3] = 1;
    Transform(position, light.lightViewMatrix);
    Transform(target, light.lightViewMatrix);
    Expect(Near(position[0], 0) && Near(position[1], 0) && Near(position[2], 0),
           "light world position maps to the view origin");
    Expect(Near(target[0], 0) && Near(target[1], 0) && target[2] < 0,
           "light target lies on the right-handed negative view Z axis");
}

void TestOmniAndFrameReplacement() {
    Scene scene("omni");
    auto source = std::make_shared<SceneObjectOmniLight>();
    auto node = AddLight(scene, source);
    std::string colorName = "light", intensityName = "intensity";
    Vector4f color{0.25f, 0.5f, 0.75f, 1};
    source->SetColor(colorName, color);
    source->SetParam(intensityName, 7.0f);
    source->SetIfCastShadow(true);
    AttenCurve distance{};
    distance.type = AttenCurveType::kInverseSquare;
    distance.u.inverse_squre_params = {2, 3, 4, 5, 6};
    source->SetDistanceAttenuation(distance);

    LightingFrame frame;
    for (bool openGL : {false, true}) {
        Expect(PrepareSceneLights(scene, openGL, frame),
               "prepare omni without an application or GPU");
        const auto& light = frame.lightInfo.lights[0];
        Expect(frame.numLights == 1 && frame.clipSpaceType == (openGL ? 0 : 1),
               "publish light count and selected clip convention");
        Expect(light.lightType == LightType::Omni &&
                   light.lightGuid == source->GetGuid() &&
                   Near(light.lightColor[2], 0.75f) &&
                   Near(light.lightIntensity, 7) && light.lightCastShadow == 1,
               "preserve light identity, color, intensity and shadow flag");
        Expect(light.lightDistAttenCurveType == distance.type &&
                   Near(light.lightDistAttenCurveParams[0][0], 2) &&
                   Near(light.lightDistAttenCurveParams[1][0], 6),
               "preserve distance attenuation parameter layout");
        Expect(Near(light.lightPosition[0], 3) &&
                   Near(light.lightPosition[1], -2) &&
                   Near(light.lightPosition[2], 6) &&
                   Near(light.lightDirection[2], -1),
               "transform position and normalized direction");
        ExpectView(light, {3, -2, 5, 1});
        ExpectDepth(light, openGL, 1, POINT_SHADOW_FAR_PLANE);
        Expect(Near(light.lightProjectionMatrix[0][0], 1) &&
                   Near(light.lightProjectionMatrix[1][1], 1),
               "omni shadow face retains square 90 degree projection");
    }
    node->MoveBy(2, 0, 0);
    node->AppendTransform(
        "rotation", std::make_shared<SceneObjectRotation>('x', float(PI / 2)));
    Expect(PrepareSceneLights(scene, false, frame),
           "read animated light transform on the next preparation");
    Expect(Near(frame.lightInfo.lights[0].lightDirection[1], 1) &&
               Near(frame.lightInfo.lights[0].lightDirection[2], 0),
           "rotate light direction without translation contamination");
    Expect(Near(frame.lightInfo.lights[0].lightPosition[0], 5),
           "runtime movement updates the light position");
    Scene empty("replacement");
    Expect(PrepareSceneLights(empty, true, frame) && frame.numLights == 0,
           "replacement scene does not retain previous lights");
}

void TestSpotAndArea() {
    Scene scene("spot");
    auto spot = std::make_shared<SceneObjectSpotLight>();
    AddLight(scene, spot);
    AttenCurve angle{};
    angle.type = AttenCurveType::kSmooth;
    angle.u.smooth_params = {PI / 8, PI / 4};
    spot->SetAngleAttenuation(angle);
    LightingFrame frame;
    for (bool openGL : {false, true}) {
        Expect(PrepareSceneLights(scene, openGL, frame), "prepare spot light");
        const auto& light = frame.lightInfo.lights[0];
        Expect(light.lightType == LightType::Spot &&
                   light.lightAngleAttenCurveType == angle.type &&
                   Near(light.lightAngleAttenCurveParams[0][0], PI / 8) &&
                   Near(light.lightAngleAttenCurveParams[0][1], PI / 4),
               "preserve angle attenuation and cone parameters");
        Expect(Near(light.lightProjectionMatrix[0][0], 1),
               "spot projection uses twice the outer attenuation angle");
        ExpectDepth(light, openGL, 1, POINT_SHADOW_FAR_PLANE);
    }
    auto area = std::make_shared<SceneObjectAreaLight>();
    area->SetDimension({4, 2});
    scene.Lights["light"] = area;
    Expect(PrepareSceneLights(scene, false, frame), "prepare area replacement");
    const auto& light = frame.lightInfo.lights[0];
    Expect(light.lightType == LightType::Area && Near(light.lightSize[0], 4) &&
               Near(light.lightSize[1], 2) &&
               light.lightAngleAttenCurveType == AttenCurveType::kNone &&
               Near(light.lightAngleAttenCurveParams[0][1], 0),
           "area replacement retains dimensions but not previous spot state");
    ExpectView(light, {3, -2, 5, 1});
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            Expect(Near(light.lightProjectionMatrix[i][j], i == j ? 1 : 0),
                   "area light retains its identity projection");
        }
    }
}

void TestDirectionalShadows() {
    Scene scene("directional");
    AddLight(scene, std::make_shared<SceneObjectInfiniteLight>());
    LightingFrame frame;
    for (bool openGL : {false, true}) {
        Expect(PrepareSceneLights(scene, openGL, frame),
               "directional light works without a scene camera");
        const auto& light = frame.lightInfo.lights[0];
        Expect(light.lightType == LightType::Infinity &&
                   Near(light.lightPosition[3], 0) &&
                   Near(light.lightPosition[2], 1000),
               "retain fallback directional placement and shader W");
        ExpectView(light, {0, 0, 0, 1});
        ExpectDepth(light, openGL, 1, 1250);
    }
    auto node = std::make_shared<SceneCameraNode>("shadow camera");
    node->AddSceneObjectRef("camera");
    node->AppendTransform(
        "position", std::make_shared<SceneObjectTranslation>(3.0f, 4.0f, 5.0f));
    scene.SceneGraph->AppendChild(node);
    scene.CameraNodes.emplace("camera", node);
    auto camera = std::make_shared<SceneObjectPerspectiveCamera>(float(PI / 3));
    std::string nearParam = "near", farParam = "far";
    camera->SetParam(nearParam, 2);
    camera->SetParam(farParam, 80);
    scene.Cameras.emplace("camera", camera);
    for (bool openGL : {false, true}) {
        Expect(PrepareSceneLights(scene, openGL, frame),
               "prepare camera-relative directional shadow");
        const auto& light = frame.lightInfo.lights[0];
        Expect(Near(light.lightPosition[0], 3) &&
                   Near(light.lightPosition[1], 4) &&
                   Near(light.lightPosition[2], 63.5f),
               "shadow target uses transformed weighted camera clipping range");
        ExpectView(light, {3, 4, -16.5f, 1});
        ExpectDepth(light, openGL, 2, 100);
        Expect(Near(light.lightProjectionMatrix[0][0], 1.0f / 20),
               "directional shadow extent uses a quarter of camera far clip");
    }
    camera->SetParam(farParam, 4000);
    Expect(PrepareSceneLights(scene, false, frame) &&
               Near(frame.lightInfo.lights[0].lightProjectionMatrix[0][0],
                    1.0f / 800),
           "directional shadow half extent remains capped at 800");
}

std::shared_ptr<SceneObjectSpotLight> ShadowSpot() {
    auto light = std::make_shared<SceneObjectSpotLight>();
    AttenCurve angle{};
    angle.type = AttenCurveType::kLinear;
    angle.u.linear_params = {PI / 8, PI / 4};
    light->SetAngleAttenuation(angle);
    light->SetIfCastShadow(true);
    return light;
}

void TestPreparedShadowLayers() {
    Scene scene("colored spots");
    auto red = ShadowSpot();
    auto green = ShadowSpot();
    std::string colorName = "light";
    Vector4f redColor{1, 0, 0, 1}, greenColor{0, 1, 0, 1};
    red->SetColor(colorName, redColor);
    green->SetColor(colorName, greenColor);
    AddLight(scene, red, "red");
    AddLight(scene, green, "green");

    LightingFrame frame;
    for (bool openGL : {false, true}) {
        for (int tick = 0; tick < 3; ++tick) {
            Expect(PrepareSceneLights(scene, openGL, frame),
                   "prepare colored spot shadows on every frame");
            std::set<int> layers;
            for (int i = 0; i < frame.numLights; ++i) {
                const auto& light = frame.lightInfo.lights[i];
                Expect(light.lightCastShadow &&
                           (light.lightGuid == red->GetGuid() ||
                            light.lightGuid == green->GetGuid()),
                       "retain each colored caster's identity");
                layers.insert(light.lightShadowMapIndex);
            }
            Expect(layers == std::set<int>{0, 1},
                   "distinct spot layers are ready before backend upload");
        }
        scene.LightNodes.rehash(scene.LightNodes.bucket_count() * 2);
    }

    scene.LightNodes.clear();
    AddLight(scene, green, "remaining");
    Expect(PrepareSceneLights(scene, false, frame) && frame.numLights == 1 &&
               frame.lightInfo.lights[0].lightShadowMapIndex == 0,
           "removing a caster rebuilds a dense shadow allocation");
}

void TestShadowCapacity() {
    Scene scene("mixed shadow pools");
    for (uint32_t i = 0; i < GfxConfiguration::kMaxShadowMapCount + 2; ++i) {
        std::shared_ptr<SceneObjectLight> source;
        if (i % 2 == 0) {
            source = ShadowSpot();
        } else {
            auto area = std::make_shared<SceneObjectAreaLight>();
            area->SetIfCastShadow(true);
            source = area;
        }
        AddLight(scene, source, "planar" + std::to_string(i));
    }
    for (uint32_t i = 0; i < GfxConfiguration::kMaxCubeShadowMapCount + 1; ++i) {
        auto source = std::make_shared<SceneObjectOmniLight>();
        source->SetIfCastShadow(true);
        AddLight(scene, source, "omni" + std::to_string(i));
    }
    for (uint32_t i = 0; i < GfxConfiguration::kMaxGlobalShadowMapCount + 1; ++i) {
        auto source = std::make_shared<SceneObjectInfiniteLight>();
        source->SetIfCastShadow(true);
        AddLight(scene, source, "sun" + std::to_string(i));
    }
    AddLight(scene, std::make_shared<SceneObjectOmniLight>(), "unshadowed");
    scene.LightNodes.emplace("expired", std::weak_ptr<SceneLightNode>{});

    LightingFrame frame;
    Expect(PrepareSceneLights(scene, false, frame) &&
               frame.numLights == static_cast<int>(
                   GfxConfiguration::kMaxShadowMapCount +
                   GfxConfiguration::kMaxCubeShadowMapCount +
                   GfxConfiguration::kMaxGlobalShadowMapCount + 5),
           "shadow overflow preserves illumination and skips expired nodes");
    std::set<int> planar, cube, global;
    int unshadowed = 0;
    for (int i = 0; i < frame.numLights; ++i) {
        const auto& light = frame.lightInfo.lights[i];
        if (!light.lightCastShadow) {
            ++unshadowed;
            continue;
        }
        auto& layers = light.lightType == LightType::Omni ? cube :
                       light.lightType == LightType::Infinity ? global : planar;
        Expect(layers.insert(light.lightShadowMapIndex).second,
               "each caster owns one layer in its texture pool");
    }
    auto dense = [](const std::set<int>& layers, uint32_t capacity) {
        if (layers.size() != capacity) return false;
        int next = 0;
        for (int layer : layers) {
            if (layer != next++) return false;
        }
        return true;
    };
    Expect(dense(planar, GfxConfiguration::kMaxShadowMapCount) &&
               dense(cube, GfxConfiguration::kMaxCubeShadowMapCount) &&
               dense(global, GfxConfiguration::kMaxGlobalShadowMapCount) &&
               unshadowed == 5,
           "spot and area share a bounded pool; cube and sun pools are independent");
}

void TestCapacityAndInvalidReferences() {
    Scene scene("capacity");
    auto source = std::make_shared<SceneObjectOmniLight>();
    auto node = AddLight(scene, source);
    for (int i = 1; i < MAX_LIGHTS; ++i) {
        scene.LightNodes.emplace("instance", node);
    }
    scene.LightNodes.emplace("expired", std::weak_ptr<SceneLightNode>{});
    LightingFrame frame;
    Expect(PrepareSceneLights(scene, false, frame) &&
               frame.numLights == MAX_LIGHTS,
           "exact capacity succeeds and expired nodes do not consume slots");
    const auto previousGuid = frame.lightInfo.lights[0].lightGuid;
    auto expectUnchanged = [&] {
        Expect(frame.numLights == MAX_LIGHTS && frame.clipSpaceType == 1 &&
                   frame.lightInfo.lights[0].lightGuid == previousGuid &&
                   Near(frame.lightInfo.lights[MAX_LIGHTS - 1].lightPosition[0],
                        3),
               "failure preserves published lighting and metadata");
    };
    scene.LightNodes.emplace("overflow", node);
    Expect(!PrepareSceneLights(scene, true, frame),
           "reject light overflow before indexing the constant buffer");
    expectUnchanged();
    scene.LightNodes.clear();
    scene.LightNodes.emplace("missing", node);
    scene.Lights.clear();
    Expect(!PrepareSceneLights(scene, true, frame),
           "reject unresolved light reference");
    expectUnchanged();
    auto spot = std::make_shared<SceneObjectSpotLight>();
    AttenCurve angle{};
    angle.type = AttenCurveType::kSmooth;
    angle.u.smooth_params = {0, std::numeric_limits<float>::quiet_NaN()};
    spot->SetAngleAttenuation(angle);
    scene.Lights.emplace("light", spot);
    Expect(!PrepareSceneLights(scene, true, frame), "reject invalid spot cone");
    expectUnchanged();
    scene.Lights["light"] = std::make_shared<SceneObjectInfiniteLight>();
    auto cameraNode = std::make_shared<SceneCameraNode>("missing camera");
    cameraNode->AddSceneObjectRef("camera");
    scene.CameraNodes.emplace("camera", cameraNode);
    Expect(!PrepareSceneLights(scene, true, frame),
           "reject unresolved directional shadow camera");
    expectUnchanged();
    auto camera = std::make_shared<SceneObjectPerspectiveCamera>(float(PI / 3));
    std::string farParam = "far";
    camera->SetParam(farParam, 0);
    scene.Cameras.emplace("camera", camera);
    Expect(!PrepareSceneLights(scene, true, frame),
           "reject invalid directional shadow clipping range");
    expectUnchanged();
}
}  // namespace

int main() {
    TestOmniAndFrameReplacement();
    TestSpotAndArea();
    TestDirectionalShadows();
    TestPreparedShadowLayers();
    TestShadowCapacity();
    TestCapacityAndInvalidReferences();
    return failures == 0 ? 0 : 1;
}
