#include <atomic>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "AssetLoader.hpp"
#include "Scene.hpp"
#include "SceneResourceLoader.hpp"

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
    explicit MemoryAssets(Buffer image) : png(std::move(image)) {}

    Buffer SyncOpenAndReadBinary(const char* filename) override {
        ++reads;
        const std::string name(filename);
        if (name == "Textures/texture_missing.png") {
            ++fallbackReads;
            if (missingFallback) return {};
        } else if (name == "missing.png" || name == "other-missing.png") {
            return {};
        }
        if (name == truncatedName) {
            // keep the signature, IHDR and a partial IDAT chunk only
            Buffer result(50);
            std::memcpy(result.GetData(), png.GetData(), 50);
            return result;
        }
        Buffer result(png.GetDataSize());
        std::memcpy(result.GetData(), png.GetData(), png.GetDataSize());
        return result;
    }
    std::string SyncOpenAndReadTextFileToString(const char*) override {
        throw std::logic_error("Texture loading must only read binary data");
    }

    Buffer png;
    std::string truncatedName;
    std::atomic<int> reads{0};
    std::atomic<int> fallbackReads{0};
    bool missingFallback = false;
};

void TestResourceLoading(MemoryAssets& assets) {
    SceneResourceLoader resources(assets);
    Scene scene("texture loading");
    scene.SkyBox.reset();
    scene.Terrain.reset();
    auto material = std::make_shared<SceneObjectMaterial>("all channels");
    for (const char* channel :
         {"diffuse", "specular", "specular_power", "emission", "opacity",
          "transparency", "normal", "metallic", "roughness", "ao", "height"}) {
        material->SetTexture(channel, std::string(channel) + ".png");
    }
    scene.Materials.emplace("material", material);
    auto shared = std::make_shared<SceneObjectTexture>("shared.png");
    scene.GetMaterial("default")->SetTexture("diffuse", shared);
    auto second = std::make_shared<SceneObjectMaterial>("shared");
    second->SetTexture("diffuse", shared);
    second->SetTexture("normal", shared);
    scene.Materials.emplace("shared", second);
    Expect(assets.reads == 0,
           "constructing scenes and assigning texture names performs no IO");
    Expect(resources.Load(scene), "load all material texture channels");
    Expect(assets.reads == 12,
           "load each texture object once, including default");
    for (const auto& texture : material->GetTextures()) {
        Expect(texture && texture->GetTextureImage(),
               "every material channel receives an image");
    }
    Expect(resources.Load(scene) && assets.reads == 12,
           "prepared textures are not loaded again");

    shared->SetName("missing.png");
    second->SetTexture("normal", "other-missing.png");
    Expect(!shared->GetTextureImage(), "renaming invalidates the old image");
    Expect(resources.Load(scene), "missing textures use the explicit fallback");
    auto image = shared->GetTextureImage();
    Expect(image && image == second->GetNormal().ValueMap->GetTextureImage() &&
               assets.fallbackReads == 1,
           "missing textures share a single fallback decode per loader");
    Expect(image && image->Width == 64 && image->Height == 64 && image->data &&
               image->pixel_format == PIXEL_FORMAT::RGBA8 &&
               image->GetR(4, 4) == 230 && image->GetG(4, 4) == 35 &&
               image->GetB(4, 4) == 155,
           "fallback retains the expected icon pixels and format");

    shared->SetName("unsupported");
    Expect(!resources.Load(scene) && !shared->GetTextureImage(),
           "unsupported format is an explicit load failure without stale data");
}

void TestTruncatedPng(MemoryAssets& assets) {
    assets.truncatedName = "truncated.png";
    SceneResourceLoader resources(assets);
    Scene scene("truncated");
    scene.SkyBox.reset();
    scene.Terrain.reset();
    scene.GetMaterial("default")->SetTexture("diffuse", "truncated.png");
    Expect(!resources.Load(scene), "truncated PNG fails resource loading");
    assets.truncatedName.clear();
}

void TestEnvironmentTextures(MemoryAssets& assets) {
    SceneResourceLoader resources(assets);
    Scene scene("environment");
    scene.SkyBox->SetName("sky", "png");
    Expect(resources.Load(scene), "prepare skybox and terrain textures");
    for (auto* texture : scene.GetTextures()) {
        Expect(texture->GetTextureImage() != nullptr,
               "environment textures all receive images");
    }
    Expect(
        scene.GetTextures().size() == scene.SkyBox->GetTextures().size() +
                                          scene.Terrain->GetTextures().size(),
        "enumeration includes every skybox face and terrain tile");
}

void TestUnavailableFallback() {
    MemoryAssets assets{Buffer()};
    assets.missingFallback = true;
    SceneResourceLoader resources(assets);
    Scene scene("missing fallback");
    scene.SkyBox.reset();
    scene.Terrain.reset();
    scene.GetMaterial("default")->SetTexture("diffuse", "missing.png");
    Expect(!resources.Load(scene),
           "unavailable fallback fails resource loading");
}
}  // namespace

int main() {
    AssetLoader files;
    if (files.Initialize() != 0) {
        std::cerr << "Failed to initialize the asset loader\n";
        return 1;
    }
    auto png = files.SyncOpenAndReadBinary("Textures/texture_missing.png");
    if (!png.GetDataSize()) {
        std::cerr << "Missing fallback test fixture\n";
        return 1;
    }
    MemoryAssets assets(std::move(png));
    TestResourceLoading(assets);
    TestTruncatedPng(assets);
    TestEnvironmentTextures(assets);
    TestUnavailableFallback();
    return failures == 0 ? 0 : 1;
}
