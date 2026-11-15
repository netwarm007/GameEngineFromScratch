#include "SceneResourceLoader.hpp"

#include <future>
#include <iostream>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ASTC.hpp"
#include "BMP.hpp"
#include "DDS.hpp"
#include "HDR.hpp"
#include "JPEG.hpp"
#include "PNG.hpp"
#include "PVR.hpp"
#include "Scene.hpp"
#include "TGA.hpp"

using namespace My;
using namespace std;

namespace {
constexpr char kMissingTexturePath[] = "Textures/texture_missing.png";

shared_ptr<Image> ParseImage(const string& name, Buffer& buffer) {
    const auto dot = name.find_last_of('.');
    const string ext = dot == string::npos ? string() : name.substr(dot);
    Image image;
    if (ext == ".jpg" || ext == ".jpeg") {
        image = JfifParser().Parse(buffer);
    } else if (ext == ".png") {
        image = PngParser().Parse(buffer);
    } else if (ext == ".bmp") {
        image = BmpParser().Parse(buffer);
    } else if (ext == ".tga") {
        image = TgaParser().Parse(buffer);
    } else if (ext == ".dds") {
        image = DdsParser().Parse(buffer);
    } else if (ext == ".hdr") {
        image = HdrParser().Parse(buffer);
    } else if (ext == ".astc") {
        image = AstcParser().Parse(buffer);
    } else if (ext == ".pvr") {
        image = PVR::PvrParser().Parse(buffer);
    } else {
        cerr << "[SceneResourceLoader] Unsupported texture format: " << name
             << endl;
        return nullptr;
    }

    if (!image.Width || !image.Height || !image.data || !image.data_size) {
        cerr << "[SceneResourceLoader] Texture decoding failed: " << name
             << endl;
        return nullptr;
    }
    return make_shared<Image>(std::move(image));
}
}  // namespace

shared_ptr<Image> SceneResourceLoader::GetFallbackImage() {
    call_once(m_FallbackOnce, [this] {
        auto buffer = m_Assets.SyncOpenAndReadBinary(kMissingTexturePath);
        if (!buffer.GetDataSize()) {
            cerr << "[SceneResourceLoader] Fallback texture unavailable: "
                 << kMissingTexturePath << endl;
            return;
        }
        m_FallbackImage = ParseImage(kMissingTexturePath, buffer);
    });
    return m_FallbackImage;
}

shared_ptr<Image> SceneResourceLoader::LoadImage(const string& name) {
    auto buffer = m_Assets.SyncOpenAndReadBinary(name.c_str());
    if (!buffer.GetDataSize()) {
        cerr << "[SceneResourceLoader] Texture unavailable: " << name
             << "; using fallback texture " << kMissingTexturePath << endl;
        return GetFallbackImage();
    }
    return ParseImage(name, buffer);
}

bool SceneResourceLoader::Load(Scene& scene) {
    vector<pair<SceneObjectTexture*, future<shared_ptr<Image>>>> pending;
    unordered_set<SceneObjectTexture*> visited;
    for (auto* texture : scene.GetTextures()) {
        if (!visited.insert(texture).second || texture->GetTextureImage()) {
            continue;
        }
        pending.emplace_back(
            texture, async(launch::async, [this, name = texture->GetName()] {
                return LoadImage(name);
            }));
    }

    bool loaded = true;
    for (auto& [texture, result] : pending) {
        auto image = result.get();
        if (!image) {
            cerr << "[SceneResourceLoader] Cannot prepare texture: "
                 << texture->GetName() << endl;
            loaded = false;
        } else {
            texture->SetTextureImage(std::move(image));
        }
    }
    return loaded;
}
