#import <MetalKit/MetalKit.h>

#import "MetalPipelineStateManager.h"
#import "MetalPipelineState.h"
#include "AssetLoader.hpp"
#include "SceneObjectVertexArray.hpp"

using namespace My;
using namespace std;

static void initMtlVertexDescriptor(MTLVertexDescriptor* mtlVertexDescriptor, MetalPipelineState* pState)
{
    // Positions
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributePosition].format = MTLVertexFormatFloat3;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributePosition].offset = 0;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributePosition].bufferIndex = VertexAttribute::VertexAttributePosition;

    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributePosition].stride = 12;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributePosition].stepRate = 1;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributePosition].stepFunction = MTLVertexStepFunctionPerVertex;

    // Normals
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeNormal].format = MTLVertexFormatFloat3;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeNormal].offset = 0;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeNormal].bufferIndex = VertexAttribute::VertexAttributeNormal;

    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeNormal].stride = 12;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeNormal].stepRate = 1;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeNormal].stepFunction = MTLVertexStepFunctionPerVertex;

    // Tangent
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTangent].format = MTLVertexFormatFloat3;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTangent].offset = 0;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTangent].bufferIndex = VertexAttribute::VertexAttributeTangent;

    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTangent].stride = 12;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTangent].stepRate = 1;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTangent].stepFunction = MTLVertexStepFunctionPerVertex;

    // UV
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTexcoord].format = MTLVertexFormatFloat2;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTexcoord].offset = 0;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTexcoord].bufferIndex = VertexAttribute::VertexAttributeTexcoord;

    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTexcoord].stride = 8;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTexcoord].stepRate = 1;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTexcoord].stepFunction = MTLVertexStepFunctionPerVertex;
}

NSString* shaderFileName2MainFuncName(std::string shaderFileName)
{
    NSString* str = [NSString stringWithCString:shaderFileName.c_str()
                              encoding: [NSString defaultCStringEncoding]];
    str = [str stringByReplacingOccurrencesOfString: @"." withString:@"_"];
    str = [str stringByAppendingString: @"_main"];
    return str;
}

bool MetalPipelineStateManager::InitializePipelineState(PipelineState** ppPipelineState)
{
    MetalPipelineState* pState = new MetalPipelineState(**ppPipelineState);

    MTLVertexDescriptor* mtlVertexDescriptor = [[MTLVertexDescriptor alloc] init];

    initMtlVertexDescriptor(mtlVertexDescriptor, pState);

    // Load all the shader files with a metallib 
    id <MTLDevice> _device = MTLCreateSystemDefaultDevice();
    NSString *libraryFile = [[NSBundle mainBundle] pathForResource:@"Main" ofType:@"metallib"];
    NSError *error = Nil;
    id <MTLLibrary> myLibrary = [_device newLibraryWithFile:libraryFile error:&error];
    if (!myLibrary) {
        NSLog(@"Library error: %@", error);
    }

    switch (pState->pipelineType)
    {
        case PIPELINE_TYPE::GRAPHIC:
        {
            // Create pipeline state
            id<MTLFunction> vertexFunction = [myLibrary newFunctionWithName:shaderFileName2MainFuncName(pState->vertexShaderName)];
            id<MTLFunction> fragmentFunction = [myLibrary newFunctionWithName:shaderFileName2MainFuncName(pState->pixelShaderName)];

            MTLRenderPipelineDescriptor *pipelineStateDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
            pipelineStateDescriptor.label = [NSString stringWithCString:pState->pipelineStateName.c_str()
                                                    encoding:[NSString defaultCStringEncoding]];
            pipelineStateDescriptor.sampleCount = 4;
            pipelineStateDescriptor.vertexFunction = vertexFunction;
            pipelineStateDescriptor.fragmentFunction = fragmentFunction;
            pipelineStateDescriptor.vertexDescriptor = mtlVertexDescriptor;
            pipelineStateDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
            pipelineStateDescriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;

            pState->mtlRenderPipelineState = 
                [_device newRenderPipelineStateWithDescriptor:pipelineStateDescriptor error:&error];
            if (!pState->mtlRenderPipelineState)
            {
                NSLog(@"Failed to created render pipeline state %@, error %@", pipelineStateDescriptor.label, error);
            }
        }
        break;
        case PIPELINE_TYPE::COMPUTE:
        {
            id<MTLFunction> compFunction = [myLibrary newFunctionWithName:shaderFileName2MainFuncName(pState->computeShaderName)];

            pState->mtlComputePipelineState =
                [_device newComputePipelineStateWithFunction:compFunction error:&error];
            if (!pState->mtlComputePipelineState)
            {
                NSLog(@"Failed to created compute pipeline state, error %@", error);
            }
        }
        break;
        default:
            assert(0);
    }

    MTLDepthStencilDescriptor *depthStateDesc = [[MTLDepthStencilDescriptor alloc] init];

    switch(pState->depthTestMode)
    {
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

    if(pState->bDepthWrite)
    {
        depthStateDesc.depthWriteEnabled = YES;
    }
    else
    {
        depthStateDesc.depthWriteEnabled = NO;
    }

    pState->depthState = [_device newDepthStencilStateWithDescriptor:depthStateDesc];

    *ppPipelineState = pState;

    return true;
}

void MetalPipelineStateManager::DestroyPipelineState(PipelineState& pipelineState)
{

}
