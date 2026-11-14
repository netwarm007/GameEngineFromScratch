#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "AnimationManager.hpp"
#include "Linear.hpp"
#include "SceneManager.hpp"

using namespace My;

namespace {
int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

class MemoryAssets : public IAssetReader {
   public:
    Buffer SyncOpenAndReadBinary(const char*) override {
        throw std::logic_error("Scene services must read text");
    }
    std::string SyncOpenAndReadTextFileToString(const char* filename) override {
        lastFilename = filename;
        ++readCount;
        return text;
    }

    std::string text = "scene data";
    std::string lastFilename;
    int readCount = 0;
};

class RecordingParser : public ISceneParser {
   public:
    std::unique_ptr<Scene> Parse(const std::string& text) override {
        lastText = text;
        ++parseCount;
        return std::move(nextScene);
    }

    std::unique_ptr<Scene> nextScene;
    std::string lastText;
    int parseCount = 0;
};

void TestLoadingAndFailedReload() {
    MemoryAssets assets;
    RecordingParser parser;
    SceneManager scenes(assets, parser);
    Expect(scenes.Initialize() == 0, "scene service initializes without an app");
    parser.nextScene = std::make_unique<Scene>("first");
    Scene* first = parser.nextScene.get();
    Expect(scenes.LoadScene("virtual.scene") == 0, "load injected scene");
    Expect(assets.lastFilename == "virtual.scene" &&
               parser.lastText == assets.text,
           "pass filename and exact text to injected dependencies");
    Expect(scenes.GetSceneForRendering().get() == first &&
               scenes.GetSceneForPhysicalSimulation().get() == first &&
               scenes.GetSceneRevision() == 1,
           "publish parsed scene and advance revision once");

    assets.text.clear();
    Expect(scenes.LoadScene("empty.scene") != 0, "reject empty scene data");
    Expect(parser.parseCount == 1, "empty data never reaches parser");
    assets.text = "invalid scene data";
    Expect(scenes.LoadScene("invalid.scene") != 0, "reject parser failure");
    Expect(scenes.GetSceneForRendering().get() == first &&
               scenes.GetSceneRevision() == 1,
           "failed reload preserves scene and revision");
    const int reads = assets.readCount;
    Expect(scenes.LoadScene(nullptr) != 0 && scenes.LoadScene("") != 0,
           "reject missing filenames");
    Expect(assets.readCount == reads, "invalid filename never reaches reader");

    parser.nextScene = std::make_unique<Scene>("second");
    Scene* second = parser.nextScene.get();
    Expect(scenes.LoadScene("second.scene") == 0, "reload injected scene");
    Expect(scenes.GetSceneForRendering().get() == second &&
               scenes.GetSceneRevision() == 2,
           "successful reload publishes replacement and next revision");
    scenes.Finalize();
}

void TestAnimationWithoutApplication() {
    MemoryAssets assets;
    RecordingParser parser;
    SceneManager scenes(assets, parser);
    AnimationManager animations(scenes);
    Expect(animations.Initialize() == 0, "animation initializes without an app");
    animations.Tick();

    auto node = std::make_shared<BaseSceneNode>("animated");
    auto translation = std::make_shared<SceneObjectTranslation>('x', 0.0f);
    node->AppendTransform("position", translation);
    auto time = std::make_shared<Linear<float, float>>(
        std::vector<float>{0.0f, 1.0f});
    auto value = std::make_shared<Linear<float, float>>(
        std::vector<float>{5.0f, 5.0f});
    auto track = std::make_shared<SceneObjectTrack>(
        translation, time, value, SceneObjectTrackType::kScalar);
    auto clip = std::make_shared<SceneObjectAnimationClip>(0);
    clip->AddTrack(track);
    node->AttachAnimationClip(0, clip);
    parser.nextScene = std::make_unique<Scene>("animated scene");
    parser.nextScene->SceneGraph = node;
    parser.nextScene->AnimatableNodes.push_back(node);
    Expect(scenes.LoadScene("animated.scene") == 0, "load animated scene");
    animations.Tick();
    Expect((*node->GetCalculatedTransform())[3][0] == 5.0f,
           "injected scene animation updates transform");

    parser.nextScene = std::make_unique<Scene>("replacement");
    Expect(scenes.LoadScene("replacement.scene") == 0, "replace animated scene");
    translation->Update(7.0f);
    animations.Tick();
    Expect((*node->GetCalculatedTransform())[3][0] == 7.0f,
           "scene replacement removes old animation clips");
    animations.Finalize();
}
}  // namespace

int main() {
    TestLoadingAndFailedReload();
    TestAnimationWithoutApplication();
    return failures == 0 ? 0 : 1;
}
