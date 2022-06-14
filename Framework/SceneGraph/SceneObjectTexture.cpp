#include "SceneObjectTexture.hpp"

#include <atomic>

using namespace My;
using namespace std;

#include "AssetLoader.hpp"
#include "BMP.hpp"
#include "DDS.hpp"
#include "HDR.hpp"
#include "JPEG.hpp"
#include "PNG.hpp"
#include "TGA.hpp"
#include "ASTC.hpp"
#include "PVR.hpp"

void SceneObjectTexture::LoadTextureAsync() {
    if (!m_asyncLoadFuture.valid()) {
        m_asyncLoadFuture =
            async(launch::async, &SceneObjectTexture::LoadTexture, this);
    }
}

bool SceneObjectTexture::LoadTexture() {
    AssetLoader assetLoader;
    if (!assetLoader.FileExists(m_Name.c_str())) return false;

    cerr << "Start async loading of " << m_Name << endl;

    Image image;
    Buffer buf = assetLoader.SyncOpenAndReadBinary(m_Name.c_str());
    string ext = m_Name.substr(m_Name.find_last_of('.'));
    if (ext == ".jpg" || ext == ".jpeg") {
        JfifParser jfif_parser;
        image = jfif_parser.Parse(buf);
    } else if (ext == ".png") {
        PngParser png_parser;
        image = png_parser.Parse(buf);
    } else if (ext == ".bmp") {
        BmpParser bmp_parser;
        image = bmp_parser.Parse(buf);
    } else if (ext == ".tga") {
        TgaParser tga_parser;
        image = tga_parser.Parse(buf);
    } else if (ext == ".dds") {
        DdsParser dds_parser;
        image = dds_parser.Parse(buf);
    } else if (ext == ".hdr") {
        HdrParser hdr_parser;
        image = hdr_parser.Parse(buf);
    } else if (ext == ".astc") {
        AstcParser astc_parser;
        image = astc_parser.Parse(buf);
    } else if (ext == ".pvr") {
        PVR::PvrParser pvr_parser;
        image = pvr_parser.Parse(buf);
    } else {
        assert(0);
    }

    // GPU does not support 24bit and 48bit textures, so adjust it
    adjust_image(image);

    cerr << "End async loading of " << m_Name << endl;

    atomic_store_explicit(&m_pImage, make_shared<Image>(std::move(image)),
                          std::memory_order_release);

    return true;
}

std::shared_ptr<Image> SceneObjectTexture::GetTextureImage() {
    if (m_asyncLoadFuture.valid()) {
        m_asyncLoadFuture.wait();
        if (m_asyncLoadFuture.get()) {
            return atomic_load_explicit(&m_pImage,
                                        std::memory_order_acquire);
        } else {
            return m_pImage;
        }
    } else {
        return m_pImage;
    }
}
