#import <MetalKit/MetalKit.h>
#import <simd/simd.h>

#import "Metal2Renderer.h"

#import "Metal2GraphicsManager.h"

#include <stack>
#include "IApplication.hpp"

#include "imgui_impl_metal.h"

using namespace My;

static MTLPixelFormat getMtlPixelFormat(const COMPRESSED_FORMAT compressed_format) {
    MTLPixelFormat format;

    switch (compressed_format) {
        case COMPRESSED_FORMAT::DXT1:
        case COMPRESSED_FORMAT::BC1:
            format = MTLPixelFormatBC1_RGBA;
            break;
        case COMPRESSED_FORMAT::DXT3:
        case COMPRESSED_FORMAT::BC2:
            format = MTLPixelFormatBC2_RGBA;
            break;
        case COMPRESSED_FORMAT::DXT5:
        case COMPRESSED_FORMAT::BC3:
            format = MTLPixelFormatBC3_RGBA;
            break;
        case COMPRESSED_FORMAT::BC4:
            format = MTLPixelFormatBC4_RUnorm;
            break;
        case COMPRESSED_FORMAT::BC5:
            format = MTLPixelFormatBC5_RGUnorm;
            break;
        case COMPRESSED_FORMAT::BC6H:
            format = MTLPixelFormatBC6H_RGBUfloat;
            break;
        case COMPRESSED_FORMAT::BC7:
            format = MTLPixelFormatBC7_RGBAUnorm;
            break;
        case COMPRESSED_FORMAT::ASTC_4x4:
            format = MTLPixelFormatASTC_4x4_sRGB;
            break;
        case COMPRESSED_FORMAT::ASTC_5x4:
            format = MTLPixelFormatASTC_5x4_sRGB;
            break;
        case COMPRESSED_FORMAT::ASTC_5x5:
            format = MTLPixelFormatASTC_5x5_sRGB;
            break;
        case COMPRESSED_FORMAT::ASTC_6x5:
            format = MTLPixelFormatASTC_6x5_sRGB;
            break;
        case COMPRESSED_FORMAT::ASTC_6x6:
            format = MTLPixelFormatASTC_6x6_sRGB;
            break;
        case COMPRESSED_FORMAT::ASTC_8x5:
            format = MTLPixelFormatASTC_8x5_sRGB;
            break;
        case COMPRESSED_FORMAT::ASTC_8x6:
            format = MTLPixelFormatASTC_8x6_sRGB;
            break;
        case COMPRESSED_FORMAT::ASTC_8x8:
            format = MTLPixelFormatASTC_8x8_sRGB;
            break;
        case COMPRESSED_FORMAT::ASTC_10x5:
            format = MTLPixelFormatASTC_10x5_sRGB;
            break;
        case COMPRESSED_FORMAT::ASTC_10x6:
            format = MTLPixelFormatASTC_10x6_sRGB;
            break;
        case COMPRESSED_FORMAT::ASTC_10x8:
            format = MTLPixelFormatASTC_10x8_sRGB;
            break;
        case COMPRESSED_FORMAT::ASTC_10x10:
            format = MTLPixelFormatASTC_10x10_sRGB;
            break;
        case COMPRESSED_FORMAT::ASTC_12x10:
            format = MTLPixelFormatASTC_12x10_sRGB;
            break;
        case COMPRESSED_FORMAT::ASTC_12x12:
            format = MTLPixelFormatASTC_12x12_sRGB;
            break;
        default:
            assert(0);
    }

    return format;
}

static MTLPixelFormat getMtlPixelFormat(const PIXEL_FORMAT pixel_format) {
    MTLPixelFormat format;

    switch (pixel_format) {
        case PIXEL_FORMAT::R8:
            format = MTLPixelFormatR8Unorm;
            break;
        case PIXEL_FORMAT::RG8:
            format = MTLPixelFormatRG8Unorm;
            break;
        case PIXEL_FORMAT::RGBA8:
            format = MTLPixelFormatRGBA8Unorm;
            break;
        case PIXEL_FORMAT::RGBA16:
            format = MTLPixelFormatRGBA16Float;
            break;
        case PIXEL_FORMAT::RGBA32:
            format = MTLPixelFormatRGBA32Float;
            break;
        case PIXEL_FORMAT::D32:
            format = MTLPixelFormatDepth32Float;
            break;
        case PIXEL_FORMAT::RG32:
            format = MTLPixelFormatRG32Float;
            break;
        default:
            assert(0);
    }

    return format;
}

static MTLPixelFormat getMtlPixelFormat(const Image& img) {
    MTLPixelFormat format;

    if (img.compressed) {
        format = getMtlPixelFormat(img.compress_format);
    } else {
        format = getMtlPixelFormat(img.pixel_format);
    }

    return format;
}

// The max number of command buffers in flight
static const NSUInteger GEFSMaxBuffersInFlight = GfxConfiguration::kMaxInFlightFrameCount;

@implementation Metal2Renderer {
    std::vector<dispatch_semaphore_t> _inFlightSemaphores;
    id<MTLCommandQueue> _graphicsQueue;
    std::vector<id<MTLCommandBuffer>> _commandBuffers;
    id<MTLCommandBuffer> _computeCommandBuffer;
    id<MTLRenderCommandEncoder> _renderEncoder;
    id<MTLComputeCommandEncoder> _computeEncoder;

    // Metal objects
    id<MTLBuffer> _uniformBuffers[GEFSMaxBuffersInFlight];
    id<MTLBuffer> _lightInfo[GEFSMaxBuffersInFlight];
    ShadowMapConstants shadow_map_constants;
    std::vector<id<MTLBuffer>> _vertexBuffers;
    std::vector<id<MTLBuffer>> _indexBuffers;
    id<MTLSamplerState> _sampler0;

    MTKView* _mtkView;
}

/// Initialize with the MetalKit view from which we'll obtain our Metal device.  We'll also use this
/// mtkView object to set the pixel format and other properties of our drawable
- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView*)mtkView
                                      device:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        _mtkView = mtkView;
        _device = device;
        _inFlightSemaphores.resize(GEFSMaxBuffersInFlight);
        for (int32_t i = 0; i < GEFSMaxBuffersInFlight; i++) {
            _inFlightSemaphores[i] = dispatch_semaphore_create(GEFSMaxBuffersInFlight);
        }
    }

    return self;
}

/// Create our metal render state objects including our shaders and render state pipeline objects
- (void)loadMetal {
    // Create and load our basic Metal state objects

    for (NSUInteger i = 0; i < GEFSMaxBuffersInFlight; i++) {
        // Create and allocate our uniform buffer object.  Indicate shared storage so that both the
        // CPU can access the buffer
        _uniformBuffers[i] = [_device newBufferWithLength:kSizePerFrameConstantBuffer
                                                  options:MTLResourceStorageModeShared];

        _uniformBuffers[i].label = [NSString stringWithFormat:@"uniformBuffer %lu", i];

        _lightInfo[i] = [_device newBufferWithLength:kSizeLightInfo
                                             options:MTLResourceStorageModeShared];

        _lightInfo[i].label = [NSString stringWithFormat:@"lightInfo %lu", i];
    }

    ////////////////////////////
    // Sampler

    MTLSamplerDescriptor* samplerDescriptor = [MTLSamplerDescriptor new];
    samplerDescriptor.minFilter = MTLSamplerMinMagFilterLinear;
    samplerDescriptor.magFilter = MTLSamplerMinMagFilterLinear;
    samplerDescriptor.mipFilter = MTLSamplerMipFilterLinear;
    samplerDescriptor.rAddressMode = MTLSamplerAddressModeRepeat;
    samplerDescriptor.sAddressMode = MTLSamplerAddressModeRepeat;
    samplerDescriptor.tAddressMode = MTLSamplerAddressModeRepeat;

    _sampler0 = [_device newSamplerStateWithDescriptor:samplerDescriptor];
    [samplerDescriptor release];

    // Create the command queue
    _graphicsQueue = [_device newCommandQueue];

    // Create command lists
    _commandBuffers.resize(GEFSMaxBuffersInFlight);
    for (NSUInteger i = 0; i < GEFSMaxBuffersInFlight; i++) {
        _commandBuffers[i] = [_graphicsQueue commandBuffer];
        _commandBuffers[i].label = [NSString stringWithFormat:@"Per Frame Command Buffer %lu", i];
    }
}

- (void)initialize {
    [self loadMetal];
    ImGui_ImplMetal_Init(_device);
}

- (void)finalize {
    ImGui_ImplMetal_Shutdown();
}

- (void)createVertexBuffer:(const SceneObjectVertexArray&)v_property_array {
    id<MTLBuffer> vertexBuffer;
    auto dataSize = v_property_array.GetDataSize();
    auto pData = v_property_array.GetData();
    vertexBuffer = [_device newBufferWithBytes:pData
                                        length:dataSize
                                       options:MTLResourceStorageModeShared];
    vertexBuffer.label = [NSString stringWithCString:v_property_array.GetAttributeName().c_str()
                                            encoding:[NSString defaultCStringEncoding]];
    _vertexBuffers.push_back(vertexBuffer);
}

- (void)createIndexBuffer:(const SceneObjectIndexArray&)index_array {
    id<MTLBuffer> indexBuffer;
    auto dataSize = index_array.GetDataSize();
    auto pData = index_array.GetData();
    indexBuffer = [_device newBufferWithBytes:pData
                                       length:dataSize
                                      options:MTLResourceStorageModeShared];
    _indexBuffers.push_back(indexBuffer);
}

- (id<MTLTexture>)createTexture:(const Image&)image {
    id<MTLTexture> texture;

    @autoreleasepool {
        MTLTextureDescriptor* textureDesc = [[MTLTextureDescriptor alloc] init];

        textureDesc.pixelFormat = getMtlPixelFormat(image);
        textureDesc.width = image.Width;
        textureDesc.height = image.Height;

        // create the texture obj
        texture = [_device newTextureWithDescriptor:textureDesc];

        // now upload the data
        MTLRegion region = {
            {0, 0, 0},                      // MTLOrigin
            {image.Width, image.Height, 1}  // MTLSize
        };

        [texture replaceRegion:region mipmapLevel:0 withBytes:image.data bytesPerRow:image.pitch];
    }

    return texture;
}

- (TextureCubeArray)createSkyBox:(const std::vector<const std::shared_ptr<My::Image>>&)images;
{
    TextureCubeArray texture_out;
    id<MTLTexture> texture;

    assert(images.size() == 18);  // 6 sky-cube + 6 irrandiance + 6 radiance

    MTLPixelFormat format = getMtlPixelFormat(*images[0]);
    auto width = images[0]->Width;
    auto height = images[0]->Height;

    @autoreleasepool {
        MTLTextureDescriptor* textureDesc = [[MTLTextureDescriptor alloc] init];

        textureDesc.textureType = MTLTextureTypeCubeArray;
        textureDesc.arrayLength = 2;
        textureDesc.pixelFormat = format;
        textureDesc.width = width;
        textureDesc.height = height;
        textureDesc.mipmapLevelCount = std::max(images[16]->mipmaps.size(), (size_t)2);

        // create the texture obj
        texture = [_device newTextureWithDescriptor:textureDesc];

        // now upload the skybox
        for (int32_t slice = 0; slice < 6; slice++) {
            assert(images[slice]->mipmaps.size() == 1);
            MTLRegion region = {
                {0, 0, 0},                                        // MTLOrigin
                {images[slice]->Width, images[slice]->Height, 1}  // MTLSize
            };

            [texture replaceRegion:region
                       mipmapLevel:0
                             slice:slice
                         withBytes:images[slice]->data
                       bytesPerRow:images[slice]->pitch
                     bytesPerImage:images[slice]->data_size];
        }

        // now upload the irradiance map as 2nd mip of skybox
        for (int32_t slice = 6; slice < 12; slice++) {
            assert(images[slice]->mipmaps.size() == 1);
            MTLRegion region = {
                {0, 0, 0},                                        // MTLOrigin
                {images[slice]->Width, images[slice]->Height, 1}  // MTLSize
            };

            [texture replaceRegion:region
                       mipmapLevel:1
                             slice:slice - 6
                         withBytes:images[slice]->data
                       bytesPerRow:images[slice]->pitch
                     bytesPerImage:images[slice]->data_size];
        }

        // now upload the radiance map 2nd cubemap
        for (int32_t slice = 12; slice < 18; slice++) {
            int level = 0;
            for (auto& mip : images[slice]->mipmaps) {
                MTLRegion region = {
                    {0, 0, 0},                  // MTLOrigin
                    {mip.Width, mip.Height, 1}  // MTLSize
                };

                [texture replaceRegion:region
                           mipmapLevel:level++
                                 slice:slice - 6
                             withBytes:images[slice]->data + mip.offset
                           bytesPerRow:mip.pitch
                         bytesPerImage:mip.data_size];
            }
        }
    }

    texture_out.handler = reinterpret_cast<TextureHandler>(texture);
    texture_out.format = format;
    texture_out.width = width;
    texture_out.height = height;
    texture_out.pixel_format = images[0]->pixel_format;
    texture_out.size = 2;
    texture_out.mips = std::max(images[16]->mipmaps.size(), (size_t)2);

    return texture_out;
}

- (void)beginFrame:(My::Frame&)frame {
    @autoreleasepool {
        // Wait to ensure only GEFSMaxBuffersInFlight are getting processed by any stage in the
        // Metal pipeline (App, Metal, Drivers, GPU, etc)
        dispatch_semaphore_wait(_inFlightSemaphores[frame.frameIndex], DISPATCH_TIME_FOREVER);

        // now fill the per frame buffers
        [self setPerFrameConstants:frame.frameContext frameIndex:frame.frameIndex];
        [self setLightInfo:frame.lightInfo frameIndex:frame.frameIndex];

        MTLRenderPassDescriptor* renderPassDescriptor = _mtkView.currentRenderPassDescriptor;
        ImGui_ImplMetal_NewFrame(renderPassDescriptor);
    }

    [_commandBuffers[frame.frameIndex] release];
    _commandBuffers[frame.frameIndex] = [_graphicsQueue commandBuffer];
    _commandBuffers[frame.frameIndex].label =
        [NSString stringWithFormat:@"Per Frame Command Buffer %d", frame.frameIndex];
}

- (void)endFrame:(Frame&)frame {
    @autoreleasepool {
        MTLRenderPassDescriptor* renderPassDescriptor = _mtkView.currentRenderPassDescriptor;
        renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionLoad;
        renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionLoad;

        id<MTLRenderCommandEncoder> renderEncoder = [_commandBuffers[frame.frameIndex]
            renderCommandEncoderWithDescriptor:renderPassDescriptor];
        renderEncoder.label = @"GuiRenderEncoder";

        auto imgui_draw_data = ImGui::GetDrawData();
        if (imgui_draw_data) {
            ImGui_ImplMetal_RenderDrawData(imgui_draw_data, _commandBuffers[frame.frameIndex],
                                           renderEncoder);
        }

        [renderEncoder endEncoding];

        [_commandBuffers[frame.frameIndex] presentDrawable:_mtkView.currentDrawable];

        // Add completion hander which signals _inFlightSemaphore when Metal and the GPU has fully
        // finished processing the commands we're encoding this frame.
        __block dispatch_semaphore_t block_sema = _inFlightSemaphores[frame.frameIndex];
        [_commandBuffers[frame.frameIndex] addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
          dispatch_semaphore_signal(block_sema);
        }];

        [_commandBuffers[frame.frameIndex] commit];
    }
}

- (void)beginPass:(Frame&)frame {
    MTLRenderPassDescriptor* renderPassDescriptor;

    if (frame.renderToTexture) {
        renderPassDescriptor = [MTLRenderPassDescriptor new];
        if (frame.enableMSAA) {
            renderPassDescriptor.colorAttachments[0].texture =
                (id<MTLTexture>)frame.colorTextures[1].handler;
            renderPassDescriptor.depthAttachment.texture =
                (id<MTLTexture>)frame.depthTexture.handler;
        }
    } else {
        // Obtain a renderPassDescriptor generated from the view's drawable textures
        renderPassDescriptor = _mtkView.currentRenderPassDescriptor;
    }

    if (renderPassDescriptor != nil) {
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(
            frame.clearColor[0], frame.clearColor[1], frame.clearColor[2], frame.clearColor[3]);
        if (frame.clearRT) {
            renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
            renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionClear;
        }
        renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
        renderPassDescriptor.depthAttachment.storeAction = MTLStoreActionDontCare;

        _renderEncoder = [_commandBuffers[frame.frameIndex]
            renderCommandEncoderWithDescriptor:renderPassDescriptor];
        _renderEncoder.label = @"Render Pass Render Encoder";
    }

    [renderPassDescriptor release];

    [_renderEncoder pushDebugGroup:@"Begin Pass"];

    MTLViewport viewport{0.0,
                         0.0,
                         static_cast<double>(frame.colorTextures[0].width),
                         static_cast<double>(frame.colorTextures[0].height),
                         0.0,
                         1.0};
    [_renderEncoder setViewport:viewport];
}

- (void)endPass:(Frame&)frame {
    [_renderEncoder popDebugGroup];
    [_renderEncoder endEncoding];
    [_renderEncoder release];
}

- (void)beginCompute {
    // Create a new command buffer for each render pass to the current drawable
    _computeCommandBuffer = [_graphicsQueue commandBuffer];
    _computeCommandBuffer.label = @"MyComputeCommand";

    _computeEncoder = [_computeCommandBuffer computeCommandEncoder];
    _computeEncoder.label = @"MyComputeEncoder";
}

- (void)endCompute {
    [_computeEncoder endEncoding];

    // Finalize rendering here & push the command buffer to the GPU
    [_computeCommandBuffer commit];
}

- (void)setPipelineState:(const MetalPipelineState&)pipelineState frameContext:(const Frame&)frame {
    switch (pipelineState.pipelineType) {
        case PIPELINE_TYPE::GRAPHIC: {
            switch (pipelineState.cullFaceMode) {
                case CULL_FACE_MODE::NONE:
                    [_renderEncoder setCullMode:MTLCullModeNone];
                    break;
                case CULL_FACE_MODE::FRONT:
                    [_renderEncoder setCullMode:MTLCullModeFront];
                    break;
                case CULL_FACE_MODE::BACK:
                    [_renderEncoder setCullMode:MTLCullModeBack];
                    break;
                default:
                    assert(0);
            }

            //[_renderEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
            [_renderEncoder setFrontFacingWinding:MTLWindingClockwise];
            [_renderEncoder setRenderPipelineState:pipelineState.mtlRenderPipelineState];
            [_renderEncoder setDepthStencilState:pipelineState.depthState];

            [_renderEncoder setVertexBuffer:_uniformBuffers[frame.frameIndex] offset:0 atIndex:10];

            [_renderEncoder setFragmentBuffer:_uniformBuffers[frame.frameIndex]
                                       offset:0
                                      atIndex:10];

            [_renderEncoder setVertexBuffer:_lightInfo[frame.frameIndex] offset:0 atIndex:12];

            [_renderEncoder setFragmentBuffer:_lightInfo[frame.frameIndex] offset:0 atIndex:12];

            switch (pipelineState.flag) {
                case PIPELINE_FLAG::SHADOW:
                    [_renderEncoder setVertexBytes:static_cast<const void*>(&shadow_map_constants)
                                            length:sizeof(ShadowMapConstants)
                                           atIndex:13];
                    break;
                case PIPELINE_FLAG::NONE:
                    break;
                case PIPELINE_FLAG::DEBUG_DRAW:
                    break;
                default:
                    assert(0);
            }

            [_renderEncoder setFragmentSamplerState:_sampler0 atIndex:0];

            if (frame.skybox.handler) {
                [_renderEncoder setFragmentTexture:(id<MTLTexture>)frame.skybox.handler atIndex:10];
            }

            if (frame.brdfLUT.handler) {
                [_renderEncoder setFragmentTexture:(id<MTLTexture>)frame.brdfLUT.handler atIndex:6];
            }
        } break;
        case PIPELINE_TYPE::COMPUTE: {
            [_computeEncoder setComputePipelineState:pipelineState.mtlComputePipelineState];
        } break;
        default:
            assert(0);
    }
}

- (void)setPerFrameConstants:(const DrawFrameContext&)context frameIndex:(const int32_t)frameIndex {
    std::memcpy(_uniformBuffers[frameIndex].contents,
                &static_cast<const PerFrameConstants&>(context), sizeof(PerFrameConstants));
}

- (void)setLightInfo:(const LightInfo&)lightInfo frameIndex:(const int32_t)frameIndex {
    std::memcpy(_lightInfo[frameIndex].contents, &lightInfo, sizeof(LightInfo));
}

- (void)drawSkyBox:(const Frame&)frame {
    // Push a debug group allowing us to identify render commands in the GPU Frame Capture tool
    [_renderEncoder pushDebugGroup:@"DrawSkyBox"];

    [_renderEncoder setVertexBytes:static_cast<const void*>(My::SceneObjectSkyBox::skyboxVertices)
                            length:sizeof(My::SceneObjectSkyBox::skyboxVertices)
                           atIndex:0];

    id<MTLBuffer> indexBuffer;
    indexBuffer = [_device newBufferWithBytes:My::SceneObjectSkyBox::skyboxIndices
                                       length:sizeof(My::SceneObjectSkyBox::skyboxIndices)
                                      options:MTLResourceStorageModeShared];

    if (indexBuffer != nil) {
        // Draw skybox
        [_renderEncoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                   indexCount:sizeof(My::SceneObjectSkyBox::skyboxIndices) /
                                              sizeof(My::SceneObjectSkyBox::skyboxIndices[0])
                                    indexType:MTLIndexTypeUInt16
                                  indexBuffer:indexBuffer
                            indexBufferOffset:0];
    }

    [indexBuffer release];

    [_renderEncoder popDebugGroup];
}

// Called whenever the view needs to render
- (void)drawBatch:(const Frame&)frame {
    // Push a debug group allowing us to identify render commands in the GPU Frame Capture tool
    [_renderEncoder pushDebugGroup:@"DrawMesh"];
    for (const auto& pDbc : frame.batchContexts) {
        [_renderEncoder setVertexBytes:pDbc->modelMatrix length:64 atIndex:11];

        const auto& dbc = dynamic_cast<const MtlDrawBatchContext&>(*pDbc);

        // Set mesh's vertex buffers
        for (uint32_t bufferIndex = 0; bufferIndex < dbc.property_count; bufferIndex++) {
            id<MTLBuffer> vertexBuffer = _vertexBuffers[dbc.property_offset + bufferIndex];
            [_renderEncoder setVertexBuffer:vertexBuffer offset:0 atIndex:bufferIndex];
        }

        // Set any textures read/sampled from our render pipeline
        if (dbc.material.diffuseMap.handler) {
            [_renderEncoder setFragmentTexture:(id<MTLTexture>)dbc.material.diffuseMap.handler
                                       atIndex:0];
        }

        if (dbc.material.normalMap.handler) {
            [_renderEncoder setFragmentTexture:(id<MTLTexture>)dbc.material.normalMap.handler
                                       atIndex:1];
        }

        if (dbc.material.metallicMap.handler) {
            [_renderEncoder setFragmentTexture:(id<MTLTexture>)dbc.material.metallicMap.handler
                                       atIndex:2];
        }

        if (dbc.material.roughnessMap.handler) {
            [_renderEncoder setFragmentTexture:(id<MTLTexture>)dbc.material.roughnessMap.handler
                                       atIndex:3];
        }

        if (dbc.material.aoMap.handler) {
            [_renderEncoder setFragmentTexture:(id<MTLTexture>)dbc.material.aoMap.handler
                                       atIndex:4];
        }

        [_renderEncoder setFragmentSamplerState:_sampler0 atIndex:0];

        // Draw our mesh
        [_renderEncoder drawIndexedPrimitives:dbc.index_mode
                                   indexCount:dbc.index_count
                                    indexType:dbc.index_type
                                  indexBuffer:_indexBuffers[dbc.index_offset]
                            indexBufferOffset:0];
    }

    [_renderEncoder popDebugGroup];
}

- (void)generateTextureArray:(Texture2DArray&)texture_array {
    id<MTLTexture> texture;
    MTLPixelFormat format = getMtlPixelFormat(texture_array.pixel_format);

    @autoreleasepool {
        MTLTextureDescriptor* textureDesc = [[MTLTextureDescriptor alloc] init];

        textureDesc.textureType = MTLTextureType2DArray;
        textureDesc.arrayLength = texture_array.size;
        textureDesc.pixelFormat = format;
        textureDesc.width = texture_array.width;
        textureDesc.height = texture_array.height;
        textureDesc.storageMode = MTLStorageModePrivate;
        textureDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

        // create the texture obj
        texture = [_device newTextureWithDescriptor:textureDesc];
    }

    texture_array.handler = reinterpret_cast<TextureHandler>(texture);
    texture_array.format = format;
}

- (void)generateCubemapArray:(TextureCubeArray&)texture_array {
    id<MTLTexture> texture;
    MTLPixelFormat format = getMtlPixelFormat(texture_array.pixel_format);

    @autoreleasepool {
        MTLTextureDescriptor* textureDesc = [[MTLTextureDescriptor alloc] init];

        textureDesc.textureType = MTLTextureTypeCubeArray;
        textureDesc.arrayLength = texture_array.size;
        textureDesc.pixelFormat = format;
        textureDesc.width = texture_array.width;
        textureDesc.height = texture_array.height;
        textureDesc.storageMode = MTLStorageModePrivate;
        textureDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

        // create the texture obj
        texture = [_device newTextureWithDescriptor:textureDesc];
    }

    texture_array.handler = reinterpret_cast<TextureHandler>(texture);
    texture_array.format = format;
}

- (void)beginShadowMap:(const int32_t)light_index
             shadowmap:(const id<MTLTexture>)shadowmap
                 width:(const uint32_t)width
                height:(const uint32_t)height
           layer_index:(const int32_t)layer_index
                 frame:(const Frame&)frame {
    MTLRenderPassDescriptor* renderPassDescriptor = [MTLRenderPassDescriptor new];
    renderPassDescriptor.colorAttachments[0] = Nil;
    renderPassDescriptor.depthAttachment.texture = shadowmap;
    renderPassDescriptor.depthAttachment.level = 0;
    renderPassDescriptor.depthAttachment.slice = layer_index;
    renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionClear;
    renderPassDescriptor.depthAttachment.storeAction = MTLStoreActionStore;

    _renderEncoder =
        [_commandBuffers[frame.frameIndex] renderCommandEncoderWithDescriptor:renderPassDescriptor];
    _renderEncoder.label = @"Offline Render Encoder";

    [renderPassDescriptor release];

    [_renderEncoder pushDebugGroup:@"BeginShadowMap"];

    MTLViewport viewport{0.0,
                         0.0,
                         static_cast<double>(shadowmap.width),
                         static_cast<double>(shadowmap.height),
                         0.0,
                         1.0};
    [_renderEncoder setViewport:viewport];

    shadow_map_constants.light_index = light_index;
    shadow_map_constants.shadowmap_layer_index = static_cast<float>(layer_index);
    shadow_map_constants.near_plane = 1.0;
    shadow_map_constants.far_plane = 100.0;
}

- (void)endShadowMap:(const id<MTLTexture>)shadowmap
         layer_index:(const int32_t)layer_index
               frame:(const Frame&)frame {
    [_renderEncoder popDebugGroup];
    [_renderEncoder endEncoding];
    [_renderEncoder release];
}

- (void)setShadowMaps:(const Frame&)frame {
    if (frame.frameContext.shadowMap.handler) {
        [_renderEncoder setFragmentTexture:(id<MTLTexture>)frame.frameContext.shadowMap.handler
                                   atIndex:7];
    }

    if (frame.frameContext.globalShadowMap.handler) {
        [_renderEncoder
            setFragmentTexture:(id<MTLTexture>)frame.frameContext.globalShadowMap.handler
                       atIndex:8];
    }

    if (frame.frameContext.cubeShadowMap.handler) {
        [_renderEncoder setFragmentTexture:(id<MTLTexture>)frame.frameContext.cubeShadowMap.handler
                                   atIndex:9];
    }
}

- (void)releaseTexture:(id<MTLTexture>)texture {
    [texture release];
}

- (void)generateTextureForWrite:(Texture2D&)texture {
    id<MTLTexture> texture_out;
    MTLPixelFormat format = getMtlPixelFormat(texture.pixel_format);

    @autoreleasepool {
        MTLTextureDescriptor* textureDesc = [MTLTextureDescriptor new];

        textureDesc.pixelFormat = format;
        textureDesc.width = texture.width;
        textureDesc.height = texture.height;
        textureDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

        // create the texture obj
        texture_out = [_device newTextureWithDescriptor:textureDesc];
    }

    texture.handler = reinterpret_cast<TextureHandler>(texture_out);
    texture.format = format;
}

- (void)bindTextureForWrite:(const id<MTLTexture>)texture atIndex:(const uint32_t)atIndex {
    [_computeEncoder setTexture:texture atIndex:atIndex];
}

- (void)dispatch:(const uint32_t)width height:(const uint32_t)height depth:(const uint32_t)depth {
    [_computeEncoder pushDebugGroup:@"dispatch"];

    // Set the compute kernel's threadgroup size
    MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);
    MTLSize threadgroupCount;

    // Calculate the number of rows and columns of threadgroups given the width of the input image
    // Ensure that you cover the entire image (or more) so you process every pixel
    threadgroupCount.width = (width + threadgroupSize.width - 1) / threadgroupSize.width;
    threadgroupCount.height = (height + threadgroupSize.height - 1) / threadgroupSize.height;
    threadgroupCount.depth = (depth + threadgroupSize.depth - 1) / threadgroupSize.depth;

    [_computeEncoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];

    [_computeEncoder popDebugGroup];
}

- (void)present {
    [_mtkView setNeedsDisplay:YES];
}

- (void)createTextureView:(Texture2D&)texture_view
            texture_array:(const TextureArrayBase&)texture_array
                    slice:(const uint32_t)slice
                      mip:(const uint32_t)mip {
    id<MTLTexture> texture = (id<MTLTexture>)texture_array.handler;
    texture_view.handler =
        (TextureHandler)[texture newTextureViewWithPixelFormat:(MTLPixelFormat)texture_array.format
                                                   textureType:MTLTextureType2D
                                                        levels:{mip, 1}
                                                        slices:{slice, 1}];
    texture_view.format = texture_array.format;
    texture_view.width = texture_array.width;
    texture_view.height = texture_array.height;
}

- (void)generateTexture:(Texture2D&)texture {
    id<MTLTexture> texture_out;
    MTLPixelFormat format = getMtlPixelFormat(texture.pixel_format);

    @autoreleasepool {
        MTLTextureDescriptor* textureDesc = [MTLTextureDescriptor new];
        textureDesc.textureType = MTLTextureType2D;
        textureDesc.width = texture.width;
        textureDesc.height = texture.height;
        textureDesc.pixelFormat = format;
        textureDesc.storageMode = MTLStorageModePrivate;
        textureDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

        // create the texture obj
        texture_out = [_device newTextureWithDescriptor:textureDesc];
    }

    texture.handler = reinterpret_cast<TextureHandler>(texture_out);
    texture.format = format;
}

@end
