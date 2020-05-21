#include "SceneObjectTexture.hpp"

#include <atomic>

using namespace My;
using namespace std;

void SceneObjectTexture::LoadTextureAsync() {
    if (!m_asyncLoadFuture.valid()) {
        m_asyncLoadFuture =
            async(launch::async, &SceneObjectTexture::LoadTexture, this);
    }
}

bool SceneObjectTexture::LoadTexture() {
    if (!g_pAssetLoader->FileExists(m_Name.c_str())) return false;

    cerr << "Start async loading of " << m_Name << endl;

    Image image;
    Buffer buf = g_pAssetLoader->SyncOpenAndReadBinary(m_Name.c_str());
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
    }

    // GPU does not support 24bit and 48bit textures, so adjust it
    if (image.bitcount == 24) {
        // DXGI does not have 24bit formats so we have to extend it to 32bit
        auto new_pitch = image.pitch / 3 * 4;
        auto data_size = (size_t)new_pitch * image.Height;
        auto* data = new uint8_t[data_size];
        uint8_t* buf;
        uint8_t* src;
        for (decltype(image.Height) row = 0; row < image.Height; row++) {
            buf = data + (ptrdiff_t)row * new_pitch;
            src = image.data + (ptrdiff_t)row * image.pitch;
            for (decltype(image.Width) col = 0; col < image.Width; col++) {
                memcpy(buf, src, 3);
                memset(buf + 3, 0x00, 1);  // set alpha to 0
                buf += 4;
                src += 3;
            }
        }

        delete[] image.data;
        image.data = data;
        image.data_size = data_size;
        image.pitch = new_pitch;
        image.bitcount = 32;

        // adjust mipmaps
        for (auto& mip : image.mipmaps) {
            mip.pitch = mip.pitch / 3 * 4;
            mip.offset = mip.offset / 3 * 4;
            mip.data_size = mip.data_size / 3 * 4;
        }
    } else if (image.bitcount == 48) {
        // DXGI does not have 48bit formats so we have to extend it to 64bit
        auto new_pitch = image.pitch / 3 * 4;
        auto data_size = new_pitch * image.Height;
        auto* data = new uint8_t[data_size];
        uint8_t* buf;
        uint8_t* src;
        for (decltype(image.Height) row = 0; row < image.Height; row++) {
            buf = data + (ptrdiff_t)row * new_pitch;
            src = image.data + (ptrdiff_t)row * image.pitch;
            for (decltype(image.Width) col = 0; col < image.Width; col++) {
                memcpy(buf, src, 6);
                memset(buf + 6, 0x00, 2);  // set alpha to 0
                buf += 8;
                src += 6;
            }
        }

        delete[] image.data;
        image.data = data;
        image.data_size = data_size;
        image.pitch = new_pitch;
        image.bitcount = 64;

        // adjust mipmaps
        for (auto& mip : image.mipmaps) {
            mip.pitch = mip.pitch / 3 * 4;
            mip.offset = mip.offset / 3 * 4;
            mip.data_size = mip.data_size / 3 * 4;
        }
    }

    cerr << "End async loading of " << m_Name << endl;

    atomic_store_explicit(&m_pImage, make_shared<Image>(std::move(image)),
                          std::memory_order::memory_order_release);

    return true;
}

std::shared_ptr<Image> SceneObjectTexture::GetTextureImage() {
    if (m_asyncLoadFuture.valid()) {
        m_asyncLoadFuture.wait();
        assert(m_asyncLoadFuture.get());
        return atomic_load_explicit(&m_pImage,
                                    std::memory_order::memory_order_acquire);
    } else {
        return m_pImage;
    }
}
