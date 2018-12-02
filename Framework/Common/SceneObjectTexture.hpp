#pragma once
#include "BaseSceneObject.hpp"
#include "geommath.hpp"
#include "JPEG.hpp"
#include "PNG.hpp"
#include "BMP.hpp"
#include "TGA.hpp"
#include "DDS.hpp"
#include "HDR.hpp"
#include "AssetLoader.hpp"

namespace My {
    class SceneObjectTexture : public BaseSceneObject
    {
        protected:
            std::string m_Name;
            uint32_t m_nTexCoordIndex;
            std::shared_ptr<Image> m_pImage;
            std::vector<Matrix4X4f> m_Transforms;

        public:
            SceneObjectTexture() : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture), m_nTexCoordIndex(0) {};
            SceneObjectTexture(const std::string& name) : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture), m_Name(name), m_nTexCoordIndex(0) {};
            SceneObjectTexture(uint32_t coord_index, std::shared_ptr<Image>& image) : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture), m_nTexCoordIndex(coord_index), m_pImage(image) {};
            SceneObjectTexture(uint32_t coord_index, std::shared_ptr<Image>&& image) : BaseSceneObject(SceneObjectType::kSceneObjectTypeTexture), m_nTexCoordIndex(coord_index), m_pImage(std::move(image)) {};
            SceneObjectTexture(SceneObjectTexture&) = default;
            SceneObjectTexture(SceneObjectTexture&&) = default;
            void AddTransform(Matrix4X4f& matrix) { m_Transforms.push_back(matrix); };
            void SetName(const std::string& name) { m_Name = name; };
            void SetName(std::string&& name) { m_Name = std::move(name); };
            const std::string& GetName() const { return m_Name; };
            void LoadTexture() {
                if (!m_pImage)
                {
                    // we should lookup if the texture has been loaded already to prevent
                    // duplicated load. This could be done in Asset Loader Manager.
                    Buffer buf = g_pAssetLoader->SyncOpenAndReadBinary(m_Name.c_str());
                    std::string ext = m_Name.substr(m_Name.find_last_of("."));
                    if (ext == ".jpg" || ext == ".jpeg")
                    {
                        JfifParser jfif_parser;
                        m_pImage = std::make_shared<Image>(jfif_parser.Parse(buf));
                    }
                    else if (ext == ".png")
                    {
                        PngParser png_parser;
                        m_pImage = std::make_shared<Image>(png_parser.Parse(buf));
                    }
                    else if (ext == ".bmp")
                    {
                        BmpParser bmp_parser;
                        m_pImage = std::make_shared<Image>(bmp_parser.Parse(buf));
                    }
                    else if (ext == ".tga")
                    {
                        TgaParser tga_parser;
                        m_pImage = std::make_shared<Image>(tga_parser.Parse(buf));
                    }
                    else if (ext == ".dds")
                    {
                        DdsParser dds_parser;
                        m_pImage = std::make_shared<Image>(dds_parser.Parse(buf));
                    }
                    else if (ext == ".hdr")
                    {
                        HdrParser hdr_parser;
                        m_pImage = std::make_shared<Image>(hdr_parser.Parse(buf));
                    }
                }
            }
        
            void AdjustTextureBitcount()
            {
                // GPU does not support 24bit and 48bit textures, so adjust it
                if (m_pImage->bitcount == 24)
                {
                    // DXGI does not have 24bit formats so we have to extend it to 32bit
                    uint32_t new_pitch = m_pImage->pitch / 3 * 4;
                    size_t data_size = new_pitch * m_pImage->Height;
                    uint8_t* data = new uint8_t[data_size];
                    uint8_t* buf;
                    uint8_t* src;
                    for (uint32_t row = 0; row < m_pImage->Height; row++) {
                        buf = data + row * new_pitch;
                        src = m_pImage->data + row * m_pImage->pitch;
                        for (uint32_t col = 0; col < m_pImage->Width; col++) {
                            memcpy(buf, src, 3);
                            memset(buf+3, 0x00, 1);  // set alpha to 0
                            buf += 4;
                            src += 3;
                        }
                    }

                    delete m_pImage->data;
                    m_pImage->data = data;
                    m_pImage->data_size = data_size;
                    m_pImage->pitch = new_pitch;
                    m_pImage->bitcount = 32;
                    
                    // adjust mipmaps
                    for (uint32_t mip = 0; mip < m_pImage->mipmap_count; mip++)
                    {
                        m_pImage->mipmaps[mip].pitch = m_pImage->mipmaps[mip].pitch / 3 * 4;
                        m_pImage->mipmaps[mip].offset = m_pImage->mipmaps[mip].offset / 3 * 4;
                        m_pImage->mipmaps[mip].data_size = m_pImage->mipmaps[mip].data_size / 3 * 4;
                    }
                }
                else if (m_pImage->bitcount == 48)
                {
                    // DXGI does not have 48bit formats so we have to extend it to 64bit
                    uint32_t new_pitch = m_pImage->pitch / 3 * 4;
                    size_t data_size = new_pitch * m_pImage->Height;
                    uint8_t* data = new uint8_t[data_size];
                    uint8_t* buf;
                    uint8_t* src;
                    for (uint32_t row = 0; row < m_pImage->Height; row++) {
                        buf = data + row * new_pitch;
                        src = m_pImage->data + row * m_pImage->pitch;
                        for (uint32_t col = 0; col < m_pImage->Width; col++) {
                            memcpy(buf, src, 6);
                            memset(buf+6, 0x00, 2); // set alpha to 0
                            buf += 8;
                            src += 6;
                        }
                    }

                    delete m_pImage->data;
                    m_pImage->data = data;
                    m_pImage->data_size = data_size;
                    m_pImage->pitch = new_pitch;
                    m_pImage->bitcount = 64;
                    
                    // adjust mipmaps
                    for (uint32_t mip = 0; mip < m_pImage->mipmap_count; mip++)
                    {
                        m_pImage->mipmaps[mip].pitch = m_pImage->mipmaps[mip].pitch / 3 * 4;
                        m_pImage->mipmaps[mip].offset = m_pImage->mipmaps[mip].offset / 3 * 4;
                        m_pImage->mipmaps[mip].data_size = m_pImage->mipmaps[mip].data_size / 3 * 4;
                    }
                }
            }

            std::shared_ptr<Image> GetTextureImage()
            { 
                if (!m_pImage)
                {
                    LoadTexture();
                }

                AdjustTextureBitcount();

                return m_pImage; 
            };

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectTexture& obj);
    };
}
