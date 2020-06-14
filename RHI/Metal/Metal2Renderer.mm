#import <MetalKit/MetalKit.h>
#import <simd/simd.h>

#import "Metal2GraphicsManager.h"
#import "Metal2Renderer.h"

#include <stack>
#include "IApplication.hpp"

#include "imgui/examples/imgui_impl_metal.h"
#include "imgui/examples/imgui_impl_osx.h"

using namespace My;

// The max number of command buffers in flight
static const NSUInteger GEFSMaxBuffersInFlight = GfxConfiguration::kMaxInFlightFrameCount;

@implementation Metal2Renderer {
    dispatch_semaphore_t _inFlightSemaphore[GEFSMaxBuffersInFlight];
    id<MTLCommandQueue> _commandQueue;
    id<MTLCommandBuffer> _commandBuffer;
    id<MTLCommandBuffer> _computeCommandBuffer;
    MTLRenderPassDescriptor* _renderPassDescriptor;
    id<MTLRenderCommandEncoder> _renderEncoder;
    id<MTLComputeCommandEncoder> _computeEncoder;

    // Metal objects
    id<MTLBuffer> _uniformBuffers[GEFSMaxBuffersInFlight];
    id<MTLBuffer> _lightInfo[GEFSMaxBuffersInFlight];
    ShadowMapConstants shadow_map_constants;
    std::vector<id<MTLBuffer>> _vertexBuffers;
    std::vector<id<MTLBuffer>> _indexBuffers;
    std::vector<id<MTLTexture>> _textures;
    std::stack<uint32_t> _texture_recycled_indexes;
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
        for (int32_t i = 0; i < GEFSMaxBuffersInFlight; i++) {
            _inFlightSemaphore[i] = dispatch_semaphore_create(GEFSMaxBuffersInFlight);
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

        _uniformBuffers[i].label = [NSString stringWithFormat:@"uniformBuffer%lu", i];

        _lightInfo[i] = [_device newBufferWithLength:kSizeLightInfo
                                             options:MTLResourceStorageModeShared];

        _lightInfo[i].label = [NSString stringWithFormat:@"lightInfo%lu", i];
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
    _commandQueue = [_device newCommandQueue];
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

static MTLPixelFormat getMtlPixelFormat(const Image& img) {
    MTLPixelFormat format;

    if (img.compressed) {
        switch (img.compress_format) {
            case "DXT1"_u32:
                format = MTLPixelFormatBC1_RGBA;
                break;
            case "DXT3"_u32:
                format = MTLPixelFormatBC3_RGBA;
                break;
            case "DXT5"_u32:
                format = MTLPixelFormatBC5_RGUnorm;
                break;
            default:
                std::cerr << img << std::endl;
                assert(0);
        }
    } else {
        switch (img.bitcount) {
            case 8:
                format = MTLPixelFormatR8Unorm;
                break;
            case 16:
                format = MTLPixelFormatRG8Unorm;
                break;
            case 32:
                format = MTLPixelFormatRGBA8Unorm;
                break;
            case 64:
                if (img.is_float) {
                    format = MTLPixelFormatRGBA16Float;
                } else {
                    format = MTLPixelFormatRGBA16Unorm;
                }
                break;
            case 128:
                if (img.is_float) {
                    format = MTLPixelFormatRGBA32Float;
                } else {
                    format = MTLPixelFormatRGBA32Uint;
                }
                break;
            default:
                assert(0);
        }
    }

    return format;
}

- (uint32_t)createTexture:(const Image&)image {
    id<MTLTexture> texture;
    MTLTextureDescriptor* textureDesc = [[MTLTextureDescriptor alloc] init];

    textureDesc.pixelFormat = getMtlPixelFormat(image);
    textureDesc.width = image.Width;
    textureDesc.height = image.Height;

    // create the texture obj
    texture = [_device newTextureWithDescriptor:textureDesc];
    [textureDesc release];

    // now upload the data
    MTLRegion region = {
        {0, 0, 0},                      // MTLOrigin
        {image.Width, image.Height, 1}  // MTLSize
    };

    [texture replaceRegion:region mipmapLevel:0 withBytes:image.data bytesPerRow:image.pitch];

    uint32_t index;
    if (!_texture_recycled_indexes.empty()) {
        index = _texture_recycled_indexes.top();
        _texture_recycled_indexes.pop();
        _textures[index] = texture;
    } else {
        index = _textures.size();
        _textures.push_back(texture);
    }

    return index;
}

- (uint32_t)createSkyBox:(const std::vector<const std::shared_ptr<My::Image>>&)images;
{
    id<MTLTexture> texture;

    assert(images.size() == 18);  // 6 sky-cube + 6 irrandiance + 6 radiance

    MTLTextureDescriptor* textureDesc = [[MTLTextureDescriptor alloc] init];

    textureDesc.textureType = MTLTextureTypeCubeArray;
    textureDesc.arrayLength = 2;
    textureDesc.pixelFormat = getMtlPixelFormat(*images[0]);
    textureDesc.width = images[0]->Width;
    textureDesc.height = images[0]->Height;
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

    uint32_t index;
    if (!_texture_recycled_indexes.empty()) {
        index = _texture_recycled_indexes.top();
        _texture_recycled_indexes.pop();
        _textures[index] = texture;
    } else {
        index = _textures.size();
        _textures.push_back(texture);
    }

    return index;
}

/// Called whenever view changes orientation or layout is changed
- (void)updateDrawableSize:(CGSize)size {
#if 0
    MTLViewport viewport {0.0, 0.0,
        static_cast<double>(size.width), static_cast<double>(size.height), 0.0, 1.0};
    [_renderEncoder setViewport:viewport];
#endif
}

- (void)beginFrame:(const My::Frame&)frame {
    // Wait to ensure only GEFSMaxBuffersInFlight are getting processed by any stage in the Metal
    // pipeline (App, Metal, Drivers, GPU, etc)
    dispatch_semaphore_wait(_inFlightSemaphore[frame.frameIndex], DISPATCH_TIME_FOREVER);

    // now fill the per frame buffers
    [self setPerFrameConstants:frame.frameContext frameIndex:frame.frameIndex];
    [self setLightInfo:frame.lightInfo frameIndex:frame.frameIndex];

    ImGui_ImplMetal_NewFrame(_mtkView.currentRenderPassDescriptor);
    ImGui_ImplOSX_NewFrame(_mtkView);
}

- (void)endFrame:(const Frame&)frame {
    // Create a new command buffer for each render pass to the current drawable
    _commandBuffer = [_commandQueue commandBuffer];
    _commandBuffer.label = @"GUI Command Buffer";
    [_commandBuffer enqueue];

    if (_renderPassDescriptor) {
        _renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionLoad;
        _renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionLoad;

        _renderEncoder = [_commandBuffer renderCommandEncoderWithDescriptor:_renderPassDescriptor];
        _renderEncoder.label = @"GuiRenderEncoder";

        ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), _commandBuffer, _renderEncoder);

        [_renderEncoder endEncoding];
    }

    [_commandBuffer presentDrawable:_mtkView.currentDrawable];

    // Add completion hander which signals _inFlightSemaphore when Metal and the GPU has fully
    // finished processing the commands we're encoding this frame.
    __block dispatch_semaphore_t block_sema = _inFlightSemaphore[frame.frameIndex];
    [_commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
      dispatch_semaphore_signal(block_sema);
    }];

    [_commandBuffer commit];
}

- (void)beginPass {
    // Create a new command buffer for each render pass to the current drawable
    _commandBuffer = [_commandQueue commandBuffer];
    _commandBuffer.label = @"Online Command Buffer";
    [_commandBuffer enqueue];

    // Obtain a renderPassDescriptor generated from the view's drawable textures
    _renderPassDescriptor = _mtkView.currentRenderPassDescriptor;

    if (_renderPassDescriptor != nil) {
        _renderPassDescriptor.colorAttachments[0].clearColor =
            MTLClearColorMake(0.2f, 0.3f, 0.4f, 1.0f);
        _renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionClear;
        _renderPassDescriptor.depthAttachment.storeAction = MTLStoreActionDontCare;

        _renderEncoder = [_commandBuffer renderCommandEncoderWithDescriptor:_renderPassDescriptor];
        _renderEncoder.label = @"MyRenderEncoder";
    }
}

- (void)endPass {
    [_renderEncoder endEncoding];

    // Finalize rendering here & push the command buffer to the GPU
    [_commandBuffer commit];
}

- (void)beginCompute {
    // Create a new command buffer for each render pass to the current drawable
    _computeCommandBuffer = [_commandQueue commandBuffer];
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

            [_renderEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
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

            auto texture_id = frame.skybox;
            if (frame.skybox >= 0) {
                [_renderEncoder setFragmentTexture:_textures[texture_id] atIndex:10];
            }

            texture_id = frame.brdfLUT;
            if (texture_id >= 0) {
                [_renderEncoder setFragmentTexture:_textures[texture_id] atIndex:6];
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

- (void)drawSkyBox {
    // Push a debug group allowing us to identify render commands in the GPU Frame Capture tool
    [_renderEncoder pushDebugGroup:@"DrawSkyBox"];

    static const float skyboxVertices[] = {
        1.0f,  1.0f,  1.0f,   // 0
        -1.0f, 1.0f,  1.0f,   // 1
        1.0f,  -1.0f, 1.0f,   // 2
        1.0f,  1.0f,  -1.0f,  // 3
        -1.0f, 1.0f,  -1.0f,  // 4
        1.0f,  -1.0f, -1.0f,  // 5
        -1.0f, -1.0f, 1.0f,   // 6
        -1.0f, -1.0f, -1.0f   // 7
    };

    [_renderEncoder setVertexBytes:static_cast<const void*>(skyboxVertices)
                            length:sizeof(skyboxVertices)
                           atIndex:0];

    static const uint16_t skyboxIndices[] = {4, 7, 5, 5, 3, 4,

                                             6, 7, 4, 4, 1, 6,

                                             5, 2, 0, 0, 3, 5,

                                             6, 1, 0, 0, 2, 6,

                                             4, 3, 0, 0, 1, 4,

                                             7, 6, 5, 5, 6, 2};

    id<MTLBuffer> indexBuffer;
    indexBuffer = [_device newBufferWithBytes:skyboxIndices
                                       length:sizeof(skyboxIndices)
                                      options:MTLResourceStorageModeShared];

    if (indexBuffer != nil) {
        // Draw skybox
        [_renderEncoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                   indexCount:sizeof(skyboxIndices) / sizeof(skyboxIndices[0])
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
        if (dbc.material.diffuseMap >= 0) {
            [_renderEncoder setFragmentTexture:_textures[dbc.material.diffuseMap] atIndex:0];
        }

        if (dbc.material.normalMap >= 0) {
            [_renderEncoder setFragmentTexture:_textures[dbc.material.normalMap] atIndex:1];
        }

        if (dbc.material.metallicMap >= 0) {
            [_renderEncoder setFragmentTexture:_textures[dbc.material.metallicMap] atIndex:2];
        }

        if (dbc.material.roughnessMap >= 0) {
            [_renderEncoder setFragmentTexture:_textures[dbc.material.roughnessMap] atIndex:3];
        }

        if (dbc.material.aoMap >= 0) {
            [_renderEncoder setFragmentTexture:_textures[dbc.material.aoMap] atIndex:4];
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

- (int32_t)generateShadowMapArray:(const uint32_t)width
                           height:(const uint32_t)height
                            count:(const uint32_t)count {
    id<MTLTexture> texture;

    MTLTextureDescriptor* textureDesc = [[MTLTextureDescriptor alloc] init];

    textureDesc.textureType = MTLTextureType2DArray;
    textureDesc.arrayLength = count;
    textureDesc.pixelFormat = MTLPixelFormatDepth32Float;
    textureDesc.width = width;
    textureDesc.height = height;
    textureDesc.storageMode = MTLStorageModePrivate;
    textureDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

    // create the texture obj
    texture = [_device newTextureWithDescriptor:textureDesc];

    uint32_t index;
    if (!_texture_recycled_indexes.empty()) {
        index = _texture_recycled_indexes.top();
        _texture_recycled_indexes.pop();
        _textures[index] = texture;
    } else {
        index = _textures.size();
        _textures.push_back(texture);
    }

    [textureDesc release];

    return static_cast<int32_t>(index);
}

- (int32_t)generateCubeShadowMapArray:(const uint32_t)width
                               height:(const uint32_t)height
                                count:(const uint32_t)count {
    id<MTLTexture> texture;

    MTLTextureDescriptor* textureDesc = [[MTLTextureDescriptor alloc] init];

    textureDesc.textureType = MTLTextureTypeCubeArray;
    textureDesc.arrayLength = count;
    textureDesc.pixelFormat = MTLPixelFormatDepth32Float;
    textureDesc.width = width;
    textureDesc.height = height;
    textureDesc.storageMode = MTLStorageModePrivate;
    textureDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

    // create the texture obj
    texture = [_device newTextureWithDescriptor:textureDesc];

    uint32_t index;
    if (!_texture_recycled_indexes.empty()) {
        index = _texture_recycled_indexes.top();
        _texture_recycled_indexes.pop();
        _textures[index] = texture;
    } else {
        index = _textures.size();
        _textures.push_back(texture);
    }

    [textureDesc release];

    return static_cast<int32_t>(index);
}

- (void)beginShadowMap:(const int32_t)light_index
             shadowmap:(const int32_t)shadowmap
                 width:(const uint32_t)width
                height:(const uint32_t)height
           layer_index:(const int32_t)layer_index
                 frame:(const Frame&)frame {
    // Create a new command buffer for each render pass to the current drawable
    _commandBuffer = [_commandQueue commandBuffer];
    _commandBuffer.label = @"Offline Command Buffer";
    [_commandBuffer enqueue];

    MTLRenderPassDescriptor* renderPassDescriptor = [MTLRenderPassDescriptor new];
    renderPassDescriptor.colorAttachments[0] = Nil;
    renderPassDescriptor.depthAttachment.texture = _textures[shadowmap];
    renderPassDescriptor.depthAttachment.level = 0;
    renderPassDescriptor.depthAttachment.slice = layer_index;
    renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionClear;
    renderPassDescriptor.depthAttachment.storeAction = MTLStoreActionStore;

    _renderEncoder = [_commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
    _renderEncoder.label = @"Offline Render Encoder";

    [renderPassDescriptor release];

    [_renderEncoder pushDebugGroup:@"BeginShadowMap"];

    MTLViewport viewport{0.0,
                         static_cast<double>(_textures[shadowmap].height),
                         static_cast<double>(_textures[shadowmap].width),
                         -static_cast<double>(_textures[shadowmap].height),
                         0.0,
                         1.0};
    [_renderEncoder setViewport:viewport];

    shadow_map_constants.light_index = light_index;
    shadow_map_constants.shadowmap_layer_index = static_cast<float>(layer_index);
    shadow_map_constants.near_plane = 1.0;
    shadow_map_constants.far_plane = 100.0;
}

- (void)endShadowMap:(const int32_t)shadowmap layer_index:(const int32_t)layer_index {
    [_renderEncoder popDebugGroup];
    [_renderEncoder endEncoding];
    [_commandBuffer commit];
}

- (void)setShadowMaps:(const Frame&)frame {
    auto texture_id = frame.frameContext.shadowMap;
    if (texture_id >= 0) {
        [_renderEncoder setFragmentTexture:_textures.at(texture_id) atIndex:7];
    }

    texture_id = frame.frameContext.globalShadowMap;
    if (texture_id >= 0) {
        [_renderEncoder setFragmentTexture:_textures.at(texture_id) atIndex:8];
    }

    texture_id = frame.frameContext.cubeShadowMap;
    if (texture_id >= 0) {
        [_renderEncoder setFragmentTexture:_textures.at(texture_id) atIndex:9];
    }
}

- (void)releaseTexture:(int32_t)texture {
    [_textures[texture] release];
    _texture_recycled_indexes.push(texture);
}

- (int32_t)generateTextureForWrite:(const uint32_t)width
                            height:(const uint32_t)height {
    id<MTLTexture> texture;
    MTLTextureDescriptor* textureDesc = [MTLTextureDescriptor new];

    textureDesc.pixelFormat = MTLPixelFormatRG32Float;
    textureDesc.width = width;
    textureDesc.height = height;
    textureDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

    // create the texture obj
    texture = [_device newTextureWithDescriptor:textureDesc];
    [textureDesc release];

    int32_t index;
    if (!_texture_recycled_indexes.empty()) {
        index = _texture_recycled_indexes.top();
        _texture_recycled_indexes.pop();
        _textures[index] = texture;
    } else {
        index = _textures.size();
        _textures.push_back(texture);
    }

    return index;
}

- (void)bindTextureForWrite:(const uint32_t)id atIndex:(const uint32_t)atIndex {
    [_computeEncoder setTexture:_textures[id] atIndex:atIndex];
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

#ifdef DEBUG
static float rect_vertices[] = {-1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f,
                                -1.0f, 1.0f,  0.0f, 1.0f, 1.0f,  0.0f};

static float rect_uv[] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};

- (void)drawTextureOverlay:(const int32_t)texture_id
                   vp_left:(const float)vp_left
                    vp_top:(const float)vp_top
                  vp_width:(const float)vp_width
                 vp_height:(const float)vp_height {
    double screenWidth = _mtkView.drawableSize.width;
    double screenHeight = _mtkView.drawableSize.height;
    double halfScreenWidth = screenWidth / 2.0;
    double halfScreenHeight = screenHeight / 2.0;

    [_renderEncoder pushDebugGroup:@"Draw Texture Overlay"];

    MTLViewport viewport{(1.0 + vp_left) * halfScreenWidth,
                         (1.0 - vp_top) * halfScreenHeight,
                         vp_width * halfScreenWidth,
                         vp_height * halfScreenHeight,
                         0.0,
                         1.0};
    [_renderEncoder setViewport:viewport];

    [_renderEncoder setVertexBytes:rect_vertices length:sizeof(rect_vertices) atIndex:0];

    [_renderEncoder setVertexBytes:rect_uv length:sizeof(rect_uv) atIndex:1];

    [_renderEncoder setFragmentTexture:_textures.at(texture_id) atIndex:0];

    [_renderEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip
                       vertexStart:0
                       vertexCount:sizeof(rect_vertices) / sizeof(rect_vertices[0]) / 3];

    viewport = {0, 0, screenWidth, screenHeight, 0.0, 1.0};
    [_renderEncoder setViewport:viewport];

    [_renderEncoder popDebugGroup];
}

- (void)drawTextureArrayOverlay:(const int32_t)texture_id
                    layer_index:(const float)layer_index
                        vp_left:(const float)vp_left
                         vp_top:(const float)vp_top
                       vp_width:(const float)vp_width
                      vp_height:(const float)vp_height {
    double screenWidth = _mtkView.drawableSize.width;
    double screenHeight = _mtkView.drawableSize.height;
    double halfScreenWidth = screenWidth / 2.0;
    double halfScreenHeight = screenHeight / 2.0;

    DebugConstants constants;

    constants.layer_index = layer_index;
    constants.mip_level = 0;

    [_renderEncoder pushDebugGroup:@"Draw Texture Array Overlay"];

    [_renderEncoder setFragmentBytes:&constants length:sizeof(DebugConstants) atIndex:13];

    MTLViewport viewport{(1.0 + vp_left) * halfScreenWidth,
                         (1.0 - vp_top) * halfScreenHeight,
                         vp_width * halfScreenWidth,
                         vp_height * halfScreenHeight,
                         0.0,
                         1.0};
    [_renderEncoder setViewport:viewport];

    [_renderEncoder setVertexBytes:rect_vertices length:sizeof(rect_vertices) atIndex:0];

    [_renderEncoder setVertexBytes:rect_uv length:sizeof(rect_uv) atIndex:1];

    [_renderEncoder setFragmentTexture:_textures.at(texture_id) atIndex:0];

    [_renderEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip
                       vertexStart:0
                       vertexCount:sizeof(rect_vertices) / sizeof(rect_vertices[0]) / 3];

    viewport = {0, 0, screenWidth, screenHeight, 0.0, 1.0};
    [_renderEncoder setViewport:viewport];

    [_renderEncoder popDebugGroup];
}

static float cubemap_unwrap_vertices[] = {
    // cell (0, 0)
    -1.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, -0.33333f, 1.0f, 0.0f, -0.33333f, 1.0f, 0.0f, -1.0f, 0.0f,
    0.0f, -0.33333f, 0.0f, 0.0f,

    // cell (0, 1)
    -0.33333f, 1.0f, 0.0f, -0.33333f, 0.0f, 0.0f, 0.33333f, 1.0f, 0.0f, 0.33333f, 1.0f, 0.0f,
    -0.33333f, 0.0f, 0.0f, 0.33333f, 0.0f, 0.0f,

    // cell (0, 2)
    0.33333f, 1.0f, 0.0f, 0.33333f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.33333f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,

    // cell (1, 0)
    -1.0f, 0.0f, 0.0f, -1.0f, -1.0f, 0.0f, -0.33333f, 0.0f, 0.0f, -0.33333f, 0.0f, 0.0f, -1.0f,
    -1.0f, 0.0f, -0.33333f, -1.0f, 0.0f,

    // cell (1, 1)
    -0.33333f, 0.0f, 0.0f, -0.33333f, -1.0f, 0.0f, 0.33333f, 0.0f, 0.0f, 0.33333f, 0.0f, 0.0f,
    -0.33333f, -1.0f, 0.0f, 0.33333f, -1.0f, 0.0f,

    // cell (1, 2)
    0.33333f, 0.0f, 0.0f, 0.33333f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.33333f,
    -1.0f, 0.0f, 1.0f, -1.0f, 0.0f};

static float cubemap_unwrap_uvw[] = {
    // back
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, -1.0f,

    // left
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f,

    // front
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, -1.0f, -1.0f,

    // right
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f,

    // top
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f,

    // bottom
    -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f};

- (void)drawCubeMapOverlay:(const int32_t)texture_id
                   vp_left:(const float)vp_left
                    vp_top:(const float)vp_top
                  vp_width:(const float)vp_width
                 vp_height:(const float)vp_height
                     level:(const float)level {
    double screenWidth = _mtkView.drawableSize.width;
    double screenHeight = _mtkView.drawableSize.height;
    double halfScreenWidth = screenWidth / 2.0;
    double halfScreenHeight = screenHeight / 2.0;

    DebugConstants constants;

    constants.mip_level = level;

    [_renderEncoder pushDebugGroup:@"Draw CubeMap Overlay"];

    [_renderEncoder setFragmentBytes:&constants length:sizeof(DebugConstants) atIndex:13];

    MTLViewport viewport{(1.0 + vp_left) * halfScreenWidth,
                         (1.0 - vp_top) * halfScreenHeight,
                         vp_width * halfScreenWidth,
                         vp_height * halfScreenHeight,
                         0.0,
                         1.0};
    [_renderEncoder setViewport:viewport];

    [_renderEncoder setVertexBytes:cubemap_unwrap_vertices
                            length:sizeof(cubemap_unwrap_vertices)
                           atIndex:0];

    [_renderEncoder setVertexBytes:cubemap_unwrap_uvw length:sizeof(cubemap_unwrap_uvw) atIndex:1];

    [_renderEncoder setFragmentTexture:_textures.at(texture_id) atIndex:0];

    [_renderEncoder
        drawPrimitives:MTLPrimitiveTypeTriangle
           vertexStart:0
           vertexCount:sizeof(cubemap_unwrap_vertices) / sizeof(cubemap_unwrap_vertices[0]) / 3];

    viewport = {0, 0, screenWidth, screenHeight, 0.0, 1.0};
    [_renderEncoder setViewport:viewport];

    [_renderEncoder popDebugGroup];
}

- (void)drawCubeMapArrayOverlay:(const int32_t)texture_id
                    layer_index:(const float)layer_index
                        vp_left:(const float)vp_left
                         vp_top:(const float)vp_top
                       vp_width:(const float)vp_width
                      vp_height:(const float)vp_height
                          level:(const float)level {
    double screenWidth = _mtkView.drawableSize.width;
    double screenHeight = _mtkView.drawableSize.height;
    double halfScreenWidth = screenWidth / 2.0;
    double halfScreenHeight = screenHeight / 2.0;

    DebugConstants constants;

    constants.layer_index = layer_index;
    constants.mip_level = level;

    [_renderEncoder pushDebugGroup:@"Draw CubeMap Array Overlay"];

    [_renderEncoder setFragmentBytes:&constants length:sizeof(DebugConstants) atIndex:13];

    MTLViewport viewport{(1.0 + vp_left) * halfScreenWidth,
                         (1.0 - vp_top) * halfScreenHeight,
                         vp_width * halfScreenWidth,
                         vp_height * halfScreenHeight,
                         0.0,
                         1.0};
    [_renderEncoder setViewport:viewport];

    [_renderEncoder setVertexBytes:cubemap_unwrap_vertices
                            length:sizeof(cubemap_unwrap_vertices)
                           atIndex:0];

    [_renderEncoder setVertexBytes:cubemap_unwrap_uvw length:sizeof(cubemap_unwrap_uvw) atIndex:1];

    [_renderEncoder setFragmentTexture:_textures.at(texture_id) atIndex:0];

    [_renderEncoder
        drawPrimitives:MTLPrimitiveTypeTriangle
           vertexStart:0
           vertexCount:sizeof(cubemap_unwrap_vertices) / sizeof(cubemap_unwrap_vertices[0]) / 3];

    viewport = {0, 0, screenWidth, screenHeight, 0.0, 1.0};
    [_renderEncoder setViewport:viewport];

    [_renderEncoder popDebugGroup];
}

#endif

@end
