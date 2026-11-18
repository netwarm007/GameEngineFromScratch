#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "AssetLoader.hpp"
#include "GuiIconLoader.hpp"

using namespace My;

namespace {
int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

void TestIconLoading() {
    const auto tempDir = std::filesystem::current_path() /
                         (".gui-icon-loader-" +
                          std::to_string(std::chrono::steady_clock::now()
                                             .time_since_epoch()
                                             .count()));
    const auto invalidIcon = tempDir / "Asset" / "Textures" / "invalid.png";
    std::filesystem::create_directories(invalidIcon.parent_path());
    bool invalidIconWritten = false;
    {
        std::ofstream file(invalidIcon, std::ios::binary);
        file.write("not a PNG", 9);
        invalidIconWritten = file.good();
    }
    Expect(invalidIconWritten, "invalid PNG fixture is written");

    AssetLoader assetLoader;
    assetLoader.ClearSearchPath();
    const auto relativeTempDir =
        std::filesystem::relative(tempDir, std::filesystem::current_path());
    const std::string searchPath = relativeTempDir.string();
    Expect(assetLoader.AddSearchPath(searchPath.c_str()),
           "temporary icon search path is added");
    Expect(assetLoader.FileExists("Textures/invalid.png"),
           "invalid PNG fixture is found by the asset loader");

    Image valid = LoadGuiIconImage("Textures/Icons/scene_empty.png");
    Expect(valid.data != nullptr && valid.Width > 0 && valid.Height > 0,
           "valid scene icon is decoded");

    Image missing = LoadGuiIconImage("Textures/Icons/not-found.png");
    Expect(missing.data == nullptr, "missing scene icon returns an empty image");

    Image invalid = LoadGuiIconImage("Textures/invalid.png");
    Expect(invalid.data == nullptr, "invalid PNG returns an empty image");

    assetLoader.ClearSearchPath();
    std::filesystem::remove_all(tempDir);
}
}  // namespace

int main() {
    TestIconLoading();
    return failures ? 1 : 0;
}
