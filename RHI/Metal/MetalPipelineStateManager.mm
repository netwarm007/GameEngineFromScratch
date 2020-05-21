#import <MetalKit/MetalKit.h>

#include "AssetLoader.hpp"
#import "MetalPipelineState.h"
#import "MetalPipelineStateManager.h"
#include "SceneObjectVertexArray.hpp"

using namespace My;
using namespace std;

static void initMtlVertexDescriptor(MTLVertexDescriptor* mtlVertexDescriptor,
                                    MetalPipelineState* pState) {
    // Positions
    MTLVertexAttributeDescriptor* vertexAttributePositionDesc = [MTLVertexAttributeDescriptor new];
    vertexAttributePositionDesc.format = MTLVertexFormatFloat3;
    vertexAttributePositionDesc.offset = 0;
    vertexAttributePositionDesc.bufferIndex = VertexAttribute::VertexAttributePosition;

    MTLVertexBufferLayoutDescriptor* vertexBufferLayoutPositionDesc =
        [MTLVertexBufferLayoutDescriptor new];
    vertexBufferLayoutPositionDesc.stride = 12;
    vertexBufferLayoutPositionDesc.stepRate = 1;
    vertexBufferLayoutPositionDesc.stepFunction = MTLVertexStepFunctionPerVertex;

    // Normals
    MTLVertexAttributeDescriptor* vertexAttributeNormalDesc = [MTLVertexAttributeDescriptor new];
    vertexAttributeNormalDesc.format = MTLVertexFormatFloat3;
    vertexAttributeNormalDesc.offset = 0;
    vertexAttributeNormalDesc.bufferIndex = VertexAttribute::VertexAttributeNormal;

    MTLVertexBufferLayoutDescriptor* vertexBufferLayoutNormalDesc =
        [MTLVertexBufferLayoutDescriptor new];
    vertexBufferLayoutNormalDesc.stride = 12;
    vertexBufferLayoutNormalDesc.stepRate = 1;
    vertexBufferLayoutNormalDesc.stepFunction = MTLVertexStepFunctionPerVertex;

    // Tangent
    MTLVertexAttributeDescriptor* vertexAttributeTangentDesc = [MTLVertexAttributeDescriptor new];
    vertexAttributeTangentDesc.format = MTLVertexFormatFloat3;
    vertexAttributeTangentDesc.offset = 0;
    vertexAttributeTangentDesc.bufferIndex = VertexAttribute::VertexAttributeTangent;

    MTLVertexBufferLayoutDescriptor* vertexBufferLayoutTangentDesc =
        [MTLVertexBufferLayoutDescriptor new];
    vertexBufferLayoutTangentDesc.stride = 12;
    vertexBufferLayoutTangentDesc.stepRate = 1;
    vertexBufferLayoutTangentDesc.stepFunction = MTLVertexStepFunctionPerVertex;

    // UV
    MTLVertexAttributeDescriptor* vertexAttributeUVDesc = [MTLVertexAttributeDescriptor new];
    vertexAttributeUVDesc.format = MTLVertexFormatFloat2;
    vertexAttributeUVDesc.offset = 0;
    vertexAttributeUVDesc.bufferIndex = VertexAttribute::VertexAttributeTexcoord;

    MTLVertexBufferLayoutDescriptor* vertexBufferLayoutUVDesc =
        [MTLVertexBufferLayoutDescriptor new];
    vertexBufferLayoutUVDesc.stride = 8;
    vertexBufferLayoutUVDesc.stepRate = 1;
    vertexBufferLayoutUVDesc.stepFunction = MTLVertexStepFunctionPerVertex;

    // UVW
    MTLVertexAttributeDescriptor* vertexAttributeUVWDesc = [MTLVertexAttributeDescriptor new];
    vertexAttributeUVWDesc.format = MTLVertexFormatFloat3;
    vertexAttributeUVWDesc.offset = 0;
    vertexAttributeUVWDesc.bufferIndex = VertexAttribute::VertexAttributeTexcoord;

    MTLVertexBufferLayoutDescriptor* vertexBufferLayoutUVWDesc =
        [MTLVertexBufferLayoutDescriptor new];
    vertexBufferLayoutUVWDesc.stride = 12;
    vertexBufferLayoutUVWDesc.stepRate = 1;
    vertexBufferLayoutUVWDesc.stepFunction = MTLVertexStepFunctionPerVertex;

    switch (pState->a2vType) {
        case A2V_TYPES::A2V_TYPES_FULL:
            mtlVertexDescriptor.attributes[VertexAttributePosition] = vertexAttributePositionDesc;
            mtlVertexDescriptor.attributes[VertexAttributeNormal] = vertexAttributeNormalDesc;
            mtlVertexDescriptor.attributes[VertexAttributeTexcoord] = vertexAttributeUVDesc;
            mtlVertexDescriptor.attributes[VertexAttributeTangent] = vertexAttributeTangentDesc;

            mtlVertexDescriptor.layouts[VertexAttributePosition] = vertexBufferLayoutPositionDesc;
            mtlVertexDescriptor.layouts[VertexAttributeNormal] = vertexBufferLayoutNormalDesc;
            mtlVertexDescriptor.layouts[VertexAttributeTexcoord] = vertexBufferLayoutUVDesc;
            mtlVertexDescriptor.layouts[VertexAttributeTangent] = vertexBufferLayoutTangentDesc;

            break;
        case A2V_TYPES::A2V_TYPES_SIMPLE:
            mtlVertexDescriptor.attributes[VertexAttributePosition] = vertexAttributePositionDesc;
            vertexAttributeUVDesc.bufferIndex = 1;
            mtlVertexDescriptor.attributes[1] = vertexAttributeUVDesc;

            mtlVertexDescriptor.layouts[VertexAttributePosition] = vertexBufferLayoutPositionDesc;
            mtlVertexDescriptor.layouts[1] = vertexBufferLayoutUVDesc;

            break;
        case A2V_TYPES::A2V_TYPES_CUBE:
            mtlVertexDescriptor.attributes[VertexAttributePosition] = vertexAttributePositionDesc;
            vertexAttributeUVWDesc.bufferIndex = 1;
            mtlVertexDescriptor.attributes[1] = vertexAttributeUVWDesc;

            mtlVertexDescriptor.layouts[VertexAttributePosition] = vertexBufferLayoutPositionDesc;
            mtlVertexDescriptor.layouts[1] = vertexBufferLayoutUVWDesc;

            break;
        case A2V_TYPES::A2V_TYPES_POS_ONLY:
            mtlVertexDescriptor.attributes[VertexAttributePosition] = vertexAttributePositionDesc;

            mtlVertexDescriptor.layouts[VertexAttributePosition] = vertexBufferLayoutPositionDesc;

            break;
        default:
            assert(0);
    }

    [vertexAttributePositionDesc release];
    [vertexAttributeNormalDesc release];
    [vertexAttributeTangentDesc release];
    [vertexAttributeUVDesc release];
    [vertexAttributeUVWDesc release];

    [vertexBufferLayoutPositionDesc release];
    [vertexBufferLayoutNormalDesc release];
    [vertexBufferLayoutTangentDesc release];
    [vertexBufferLayoutUVDesc release];
    [vertexBufferLayoutUVWDesc release];
}

NSString* shaderFileName2MainFuncName(std::string shaderFileName) {
    NSString* str = [NSString stringWithCString:shaderFileName.c_str()
                                       encoding:[NSString defaultCStringEncoding]];
    str = [str stringByReplacingOccurrencesOfString:@"." withString:@"_"];
    str = [str stringByAppendingString:@"_main"];
    return str;
}

bool MetalPipelineStateManager::InitializePipelineState(PipelineState** ppPipelineState) {
    MetalPipelineState* pState = new MetalPipelineState(**ppPipelineState);

    // Load all the shader files with a metallib
    id<MTLDevice> _device = MTLCreateSystemDefaultDevice();
    NSError* error = Nil;
    NSString* libraryFile = [[NSBundle mainBundle] pathForResource:@"Main" ofType:@"metallib"];
    if (!libraryFile) {
        NSLog(@"Metal Shader Library missing");
        return false;
    }

    id<MTLLibrary> myLibrary = [_device newLibraryWithFile:libraryFile error:&error];
    if (!myLibrary) {
        NSLog(@"Library error: %@", error);
        return false;
    }

    switch (pState->pipelineType) {
        case PIPELINE_TYPE::GRAPHIC: {
            MTLVertexDescriptor* mtlVertexDescriptor = [MTLVertexDescriptor new];

            initMtlVertexDescriptor(mtlVertexDescriptor, pState);

            // Create pipeline state
            id<MTLFunction> vertexFunction = [myLibrary
                newFunctionWithName:shaderFileName2MainFuncName(pState->vertexShaderName)];
            id<MTLFunction> fragmentFunction = [myLibrary
                newFunctionWithName:shaderFileName2MainFuncName(pState->pixelShaderName)];

            MTLRenderPipelineDescriptor* pipelineStateDescriptor =
                [MTLRenderPipelineDescriptor new];
            pipelineStateDescriptor.label =
                [NSString stringWithCString:pState->pipelineStateName.c_str()
                                   encoding:[NSString defaultCStringEncoding]];
            pipelineStateDescriptor.sampleCount = pState->sampleCount;
            pipelineStateDescriptor.vertexFunction = vertexFunction;
            [vertexFunction release];
            pipelineStateDescriptor.fragmentFunction = fragmentFunction;
            [fragmentFunction release];
            pipelineStateDescriptor.vertexDescriptor = mtlVertexDescriptor;
            [mtlVertexDescriptor release];
            switch (pState->pixelFormat) {
                case PIXEL_FORMAT::INVALID:
                    pipelineStateDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatInvalid;
                    break;
                case PIXEL_FORMAT::BGRA8UNORM:
                    pipelineStateDescriptor.colorAttachments[0].pixelFormat =
                        MTLPixelFormatBGRA8Unorm;
                    break;
                default:
                    assert(0);
            }
            pipelineStateDescriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

            pState->mtlRenderPipelineState =
                [_device newRenderPipelineStateWithDescriptor:pipelineStateDescriptor error:&error];
            if (!pState->mtlRenderPipelineState) {
                NSLog(@"Failed to created render pipeline state %@, error %@",
                      pipelineStateDescriptor.label, error);
            }
            [pipelineStateDescriptor release];
        } break;
        case PIPELINE_TYPE::COMPUTE: {
            id<MTLFunction> compFunction = [myLibrary
                newFunctionWithName:shaderFileName2MainFuncName(pState->computeShaderName)];

            if (compFunction != nil) {
                pState->mtlComputePipelineState =
                    [_device newComputePipelineStateWithFunction:compFunction error:&error];
                if (!pState->mtlComputePipelineState) {
                    NSLog(@"Failed to created compute pipeline state, error %@", error);
                }

                [compFunction release];
            }
        } break;
        default:
            assert(0);
    }

    [myLibrary release];

    MTLDepthStencilDescriptor* depthStateDesc = [MTLDepthStencilDescriptor new];

    switch (pState->depthTestMode) {
        case DEPTH_TEST_MODE::NONE:
            depthStateDesc.depthCompareFunction = MTLCompareFunctionAlways;
            break;
        case DEPTH_TEST_MODE::LARGE:
            depthStateDesc.depthCompareFunction = MTLCompareFunctionGreater;
            break;
        case DEPTH_TEST_MODE::LARGE_EQUAL:
            depthStateDesc.depthCompareFunction = MTLCompareFunctionGreaterEqual;
            break;
        case DEPTH_TEST_MODE::LESS:
            depthStateDesc.depthCompareFunction = MTLCompareFunctionLess;
            break;
        case DEPTH_TEST_MODE::LESS_EQUAL:
            depthStateDesc.depthCompareFunction = MTLCompareFunctionLessEqual;
            break;
        case DEPTH_TEST_MODE::EQUAL:
            depthStateDesc.depthCompareFunction = MTLCompareFunctionEqual;
            break;
        case DEPTH_TEST_MODE::NOT_EQUAL:
            depthStateDesc.depthCompareFunction = MTLCompareFunctionNotEqual;
            break;
        case DEPTH_TEST_MODE::NEVER:
            depthStateDesc.depthCompareFunction = MTLCompareFunctionNever;
            break;
        case DEPTH_TEST_MODE::ALWAYS:
            depthStateDesc.depthCompareFunction = MTLCompareFunctionAlways;
            break;
        default:
            assert(0);
    }

    if (pState->bDepthWrite) {
        depthStateDesc.depthWriteEnabled = YES;
    } else {
        depthStateDesc.depthWriteEnabled = NO;
    }

    pState->depthState = [_device newDepthStencilStateWithDescriptor:depthStateDesc];
    [depthStateDesc release];

    [_device release];

    *ppPipelineState = pState;

    return true;
}

void MetalPipelineStateManager::DestroyPipelineState(PipelineState& pipelineState) {
    MetalPipelineState* pState = dynamic_cast<MetalPipelineState*>(&pipelineState);
    [pState->mtlRenderPipelineState release];
    [pState->mtlComputePipelineState release];
    [pState->depthState release];
}
