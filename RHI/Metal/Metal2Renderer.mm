#import <simd/simd.h>
#import <MetalKit/MetalKit.h>

#import "Metal2Renderer.h"
#import "MetalShaderManager.h"
#import "Metal2GraphicsManager.h"

#include "IApplication.hpp"

using namespace My;

// The max number of command buffers in flight
static const NSUInteger GEFSMaxBuffersInFlight = GfxConfiguration::kMaxInFlightFrameCount;

@implementation Metal2Renderer
{
    dispatch_semaphore_t _inFlightSemaphore;
    id<MTLCommandQueue> _commandQueue;
    id<MTLCommandBuffer> _commandBuffer;
    MTLRenderPassDescriptor* _renderPassDescriptor;
    id<MTLRenderCommandEncoder> _renderEncoder;

    // Metal objects
    id<MTLRenderPipelineState> _pipelineState;
    id<MTLDepthStencilState> _depthState;
    id<MTLTexture> _baseColorMap;
    id<MTLTexture> _normalMap;
    id<MTLTexture> _specularMap;
    id<MTLBuffer> _uniformBuffers[GEFSMaxBuffersInFlight];
    std::vector<id<MTLBuffer>> _vertexBuffers;
    std::vector<id<MTLBuffer>> _indexBuffers;
    id<MTLSamplerState> _sampler0;

    // The index in uniform buffers in _dynamicUniformBuffers to use for the current frame
    uint32_t _currentBufferIndex;

    // Vertex descriptor specifying how vertices will by laid out for input into our render
    // pipeline and how ModelIO should layout vertices
    MTLVertexDescriptor* _mtlVertexDescriptor;

    MTKView* _mtkView;
}

/// Initialize with the MetalKit view from which we'll obtain our Metal device.  We'll also use this
/// mtkView object to set the pixel format and other properties of our drawable
- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView device:(id<MTLDevice>) device
{
    self = [super init];
    if(self)
    {
        _currentBufferIndex = 0;
        _mtkView = mtkView;
        _device = device;
        _inFlightSemaphore = dispatch_semaphore_create(GEFSMaxBuffersInFlight);
        [self loadMetal];
    }

    return self;
}

/// Create our metal render state objects including our shaders and render state pipeline objects
- (void) loadMetal
{

    NSError *error = Nil;
    // Create and load our basic Metal state objects

    // Load all the shader files with a metallib 
    NSString *libraryFile = [[NSBundle mainBundle] pathForResource:@"Editor" ofType:@"metallib"];
    id <MTLLibrary> myLibrary = [_device newLibraryWithFile:libraryFile error:&error];
    if (!myLibrary) {
        NSLog(@"Library error: %@", error);
    }

    for(NSUInteger i = 0; i < GEFSMaxBuffersInFlight; i++)
    {
        // Create and allocate our uniform buffer object.  Indicate shared storage so that both the
        //  CPU can access the buffer
        _uniformBuffers[i] = [_device newBufferWithLength:kSizePerBatchConstantBuffer + kSizePerFrameConstantBuffer * GfxConfiguration::kMaxSceneObjectCount
                                                     options:MTLResourceStorageModeShared];

        _uniformBuffers[i].label = [NSString stringWithFormat:@"uniformBuffer%lu", i];
    }

    _mtlVertexDescriptor = [MTLVertexDescriptor new];

    // Positions.
    _mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributePosition].format = MTLVertexFormatFloat3;
    _mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributePosition].offset = 0;
    _mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributePosition].bufferIndex = 0;

    // Texture coordinates.
    _mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTexcoord].format = MTLVertexFormatFloat2;
    _mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTexcoord].offset = 0;
    _mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTexcoord].bufferIndex = 1;

    // Normals.
    _mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeNormal].format = MTLVertexFormatFloat3;
    _mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeNormal].offset = 0;
    _mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeNormal].bufferIndex = 2;

    // Tangents
    _mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTangent].format = MTLVertexFormatFloat3;
    _mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTangent].offset = 0;
    _mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTangent].bufferIndex = 3;

    // Bitangents
    _mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeBitangent].format = MTLVertexFormatFloat3;
    _mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeBitangent].offset = 0;
    _mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeBitangent].bufferIndex = 4;

    // Position Buffer Layout
    _mtlVertexDescriptor.layouts[0].stride = 12;
    _mtlVertexDescriptor.layouts[0].stepRate = 1;
    _mtlVertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;

    // UV Buffer Layout
    _mtlVertexDescriptor.layouts[1].stride = 8;
    _mtlVertexDescriptor.layouts[1].stepRate = 1;
    _mtlVertexDescriptor.layouts[1].stepFunction = MTLVertexStepFunctionPerVertex;

    // Normal Buffer Layout
    _mtlVertexDescriptor.layouts[2].stride = 12;
    _mtlVertexDescriptor.layouts[2].stepRate = 1;
    _mtlVertexDescriptor.layouts[2].stepFunction = MTLVertexStepFunctionPerVertex;

    // Tangent Buffer Layout
    _mtlVertexDescriptor.layouts[3].stride = 12;
    _mtlVertexDescriptor.layouts[3].stepRate = 1;
    _mtlVertexDescriptor.layouts[3].stepFunction = MTLVertexStepFunctionPerVertex;

    // Bitangent Buffer Layout
    _mtlVertexDescriptor.layouts[4].stride = 12;
    _mtlVertexDescriptor.layouts[4].stepRate = 1;
    _mtlVertexDescriptor.layouts[4].stepFunction = MTLVertexStepFunctionPerVertex;

    MTLSamplerDescriptor* samplerDescriptor = [MTLSamplerDescriptor new];
    samplerDescriptor.minFilter = MTLSamplerMinMagFilterNearest;
    samplerDescriptor.magFilter = MTLSamplerMinMagFilterLinear;
    samplerDescriptor.sAddressMode = MTLSamplerAddressModeRepeat;
    samplerDescriptor.tAddressMode = MTLSamplerAddressModeRepeat;

    _sampler0 = [_device newSamplerStateWithDescriptor:samplerDescriptor];

    id<MTLFunction> vertexFunction = [myLibrary newFunctionWithName:@"basic_vert_main"];
    id<MTLFunction> fragmentFunction = [myLibrary newFunctionWithName:@"basic_frag_main"];

    // Create a reusable pipeline state
    MTLRenderPipelineDescriptor *pipelineStateDescriptor = [MTLRenderPipelineDescriptor new];
    pipelineStateDescriptor.label = @"MyPipeline";
    pipelineStateDescriptor.sampleCount = _mtkView.sampleCount;
    pipelineStateDescriptor.vertexFunction = vertexFunction;
    pipelineStateDescriptor.fragmentFunction = fragmentFunction;
    pipelineStateDescriptor.vertexDescriptor = _mtlVertexDescriptor;
    pipelineStateDescriptor.colorAttachments[0].pixelFormat = _mtkView.colorPixelFormat;
    pipelineStateDescriptor.depthAttachmentPixelFormat = _mtkView.depthStencilPixelFormat;
    //pipelineStateDescriptor.stencilAttachmentPixelFormat = _mtkView.depthStencilPixelFormat;

    _pipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineStateDescriptor error:&error];
    if (!_pipelineState)
    {
        NSLog(@"Failed to created pipeline state, error %@", error);
        assert(0);
    }

    MTLDepthStencilDescriptor *depthStateDesc = [MTLDepthStencilDescriptor new];
    depthStateDesc.depthCompareFunction = MTLCompareFunctionLess;
    depthStateDesc.depthWriteEnabled = YES;
    _depthState = [_device newDepthStencilStateWithDescriptor:depthStateDesc];

    // Create the command queue
    _commandQueue = [_device newCommandQueue];
}

- (void)createVertexBuffer:(const SceneObjectVertexArray&)v_property_array
{
    id<MTLBuffer> vertexBuffer;
    auto dataSize = v_property_array.GetDataSize();
	auto pData = v_property_array.GetData();
    vertexBuffer = [_device newBufferWithBytes:pData length:dataSize options:MTLResourceStorageModeShared];
    _vertexBuffers.push_back(vertexBuffer);
}

- (void)createIndexBuffer:(const SceneObjectIndexArray&)index_array
{
    id<MTLBuffer> indexBuffer;
    auto dataSize = index_array.GetDataSize();
	auto pData = index_array.GetData();
    indexBuffer = [_device newBufferWithBytes:pData length:dataSize options:MTLResourceStorageModeShared];
    _indexBuffers.push_back(indexBuffer);
}

/// Called whenever view changes orientation or layout is changed
- (void)updateDrawableSize:(CGSize)size
{
#if 0
    /// React to resize of our draw rect.  In particular update our perspective matrix
    // Update the aspect ratio and projection matrix since the view orientation or size has changed
    float aspect = size.width / (float)size.height;

    _projectionMatrix = matrix_perspective_left_hand(65.0f * (M_PI / 180.0f), aspect, 0.1f, 100.0);
#endif
}

- (void)beginFrame
{
    // Wait to ensure only GEFSMaxBuffersInFlight are getting processed by any stage in the Metal
    // pipeline (App, Metal, Drivers, GPU, etc)
    dispatch_semaphore_wait(_inFlightSemaphore, DISPATCH_TIME_FOREVER);

    // Create a new command buffer for each render pass to the current drawable
    _commandBuffer = [_commandQueue commandBuffer];
    _commandBuffer.label = @"myCommand";

    // Obtain a renderPassDescriptor generated from the view's drawable textures
    _renderPassDescriptor = _mtkView.currentRenderPassDescriptor;

    if(_renderPassDescriptor != nil)
    {
        _renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.2f, 0.3f, 0.4f, 1.0f);

        _renderEncoder =
            [_commandBuffer renderCommandEncoderWithDescriptor:_renderPassDescriptor];
        _renderEncoder.label = @"MyRenderEncoder";

        [_renderEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
        [_renderEncoder setCullMode:MTLCullModeBack];
        [_renderEncoder setRenderPipelineState:_pipelineState];
        [_renderEncoder setDepthStencilState:_depthState];
    }
    else
    {
        assert(0);
    }
}

- (void)endFrame
{
    if(_renderPassDescriptor != nil)
    {
        [_renderEncoder endEncoding];
    }

    [_commandBuffer presentDrawable:_mtkView.currentDrawable];

    // Add completion hander which signals _inFlightSemaphore when Metal and the GPU has fully
    // finished processing the commands we're encoding this frame.
    __block dispatch_semaphore_t block_sema = _inFlightSemaphore;
    [_commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer)
     {
         dispatch_semaphore_signal(block_sema);
     }];

    // Finalize rendering here & push the command buffer to the GPU
    [_commandBuffer commit];

    _currentBufferIndex = (_currentBufferIndex + 1) % GEFSMaxBuffersInFlight;
}

- (void)setPerFrameConstants:(const DrawFrameContext&)context
{
    std::memcpy(_uniformBuffers[_currentBufferIndex].contents, 
            &static_cast<const PerFrameConstants&>(context), sizeof(PerFrameConstants));
}

- (void)setPerBatchConstants:(const DrawBatchContext&)context
{
    std::memcpy(reinterpret_cast<uint8_t*>(_uniformBuffers[_currentBufferIndex].contents) 
            + kSizePerFrameConstantBuffer + context.batchIndex * kSizePerBatchConstantBuffer
            , &static_cast<const PerBatchConstants&>(context), sizeof(PerBatchConstants));
}

// Called whenever the view needs to render
- (void)drawBatch:(const MtlDrawBatchContext&)dbc
{
    // If we've gotten a renderPassDescriptor we can render to the drawable, otherwise we'll skip
    // any rendering this frame because we have no drawable to draw to
    if(_renderPassDescriptor != nil)
    {
        // Push a debug group allowing us to identify render commands in the GPU Frame Capture tool
        [_renderEncoder pushDebugGroup:@"DrawMesh"];

        [_renderEncoder setVertexBuffer:_uniformBuffers[_currentBufferIndex]
                                  offset:0
                                 atIndex:10];

        [_renderEncoder setVertexBuffer:_uniformBuffers[_currentBufferIndex]
                                  offset:kSizePerFrameConstantBuffer + dbc.batchIndex * kSizePerBatchConstantBuffer
                                 atIndex:11];

        [_renderEncoder setFragmentBuffer:_uniformBuffers[_currentBufferIndex]
                                  offset:0
                                 atIndex:10];

        // Set mesh's vertex buffers
        for (uint32_t bufferIndex = 0; bufferIndex < dbc.property_count; bufferIndex++)
        {
            id<MTLBuffer> vertexBuffer = _vertexBuffers[bufferIndex];
            [_renderEncoder setVertexBuffer:vertexBuffer
                                    offset:0
                                   atIndex:bufferIndex];
        }

        [_renderEncoder setFragmentSamplerState:_sampler0 atIndex:0];
#if 0
        // Set any textures read/sampled from our render pipeline
        [_renderEncoder setFragmentTexture:_baseColorMap
                                  atIndex:TextureIndex::TextureIndexBaseColor];

        [_renderEncoder setFragmentTexture:_normalMap
                                  atIndex:TextureIndex::TextureIndexNormal];

        [_renderEncoder setFragmentTexture:_specularMap
                                  atIndex:TextureIndex::TextureIndexSpecular];

#endif
        // Draw our mesh
        [_renderEncoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                  indexCount:dbc.index_count
                                   indexType:MTLIndexTypeUInt32
                                 indexBuffer:_indexBuffers[dbc.index_offset]
                           indexBufferOffset:0];

        [_renderEncoder popDebugGroup];
    }
}

@end
