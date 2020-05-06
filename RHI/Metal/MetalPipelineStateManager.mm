#import <MetalKit/MetalKit.h>

#import "MetalPipelineStateManager.h"
#import "MetalPipelineState.h"
#include "AssetLoader.hpp"
#include "SceneObjectVertexArray.hpp"

using namespace My;
using namespace std;

bool MetalPipelineStateManager::InitializePipelineState(PipelineState** ppPipelineState)
{
    MetalPipelineState* pState = new MetalPipelineState(**ppPipelineState);

    MTLVertexDescriptor* mtlVertexDescriptor = [[MTLVertexDescriptor alloc] init];

    // Positions
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributePosition].format = MTLVertexFormatFloat4;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributePosition].offset = 0;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributePosition].bufferIndex = VertexAttribute::VertexAttributePosition;

    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributePosition].stride = 16;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributePosition].stepRate = 1;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributePosition].stepFunction = MTLVertexStepFunctionPerVertex;

    // Normals
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeNormal].format = MTLVertexFormatFloat4;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeNormal].offset = 0;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeNormal].bufferIndex = VertexAttribute::VertexAttributeNormal;

    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeNormal].stride = 16;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeNormal].stepRate = 1;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeNormal].stepFunction = MTLVertexStepFunctionPerVertex;

    // Normals World
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeNormalWorld].format = MTLVertexFormatFloat4;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeNormalWorld].offset = 0;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeNormalWorld].bufferIndex = VertexAttribute::VertexAttributeNormalWorld;

    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeNormalWorld].stride = 16;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeNormalWorld].stepRate = 1;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeNormalWorld].stepFunction = MTLVertexStepFunctionPerVertex;

    // View
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeView].format = MTLVertexFormatFloat4;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeView].offset = 0;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeView].bufferIndex = VertexAttribute::VertexAttributeView;

    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeView].stride = 16;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeView].stepRate = 1;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeView].stepFunction = MTLVertexStepFunctionPerVertex;

    // View World
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeViewWorld].format = MTLVertexFormatFloat4;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeViewWorld].offset = 0;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeViewWorld].bufferIndex = VertexAttribute::VertexAttributeViewWorld;

    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeViewWorld].stride = 16;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeViewWorld].stepRate = 1;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeViewWorld].stepFunction = MTLVertexStepFunctionPerVertex;

    // Tangent
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTangent].format = MTLVertexFormatFloat3;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTangent].offset = 0;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTangent].bufferIndex = VertexAttribute::VertexAttributeTangent;

    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTangent].stride = 12;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTangent].stepRate = 1;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTangent].stepFunction = MTLVertexStepFunctionPerVertex;

    // Cam Pos Tangent
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeCamPosTangent].format = MTLVertexFormatFloat3;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeCamPosTangent].offset = 0;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeCamPosTangent].bufferIndex = VertexAttribute::VertexAttributeCamPosTangent;

    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeCamPosTangent].stride = 12;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeCamPosTangent].stepRate = 1;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeCamPosTangent].stepFunction = MTLVertexStepFunctionPerVertex;

    // UV
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTexcoord].format = MTLVertexFormatFloat2;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTexcoord].offset = 0;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTexcoord].bufferIndex = VertexAttribute::VertexAttributeTexcoord;

    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTexcoord].stride = 8;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTexcoord].stepRate = 1;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTexcoord].stepFunction = MTLVertexStepFunctionPerVertex;

    // TBN 0
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTBN].format = MTLVertexFormatFloat3;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTBN].offset = 0;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTBN].bufferIndex = VertexAttribute::VertexAttributeTBN;

    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTBN].stride = 12;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTBN].stepRate = 1;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTBN].stepFunction = MTLVertexStepFunctionPerVertex;

    // TBN 1
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTBN + 1].format = MTLVertexFormatFloat3;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTBN + 1].offset = 0;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTBN + 1].bufferIndex = VertexAttribute::VertexAttributeTBN + 1;

    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTBN + 1].stride = 12;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTBN + 1].stepRate = 1;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTBN + 1].stepFunction = MTLVertexStepFunctionPerVertex;

    // TBN 2
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTBN + 2].format = MTLVertexFormatFloat3;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTBN + 2].offset = 0;
    mtlVertexDescriptor.attributes[VertexAttribute::VertexAttributeTBN + 2].bufferIndex = VertexAttribute::VertexAttributeTBN + 2;

    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTBN + 2].stride = 12;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTBN + 2].stepRate = 1;
    mtlVertexDescriptor.layouts[VertexAttribute::VertexAttributeTBN + 2].stepFunction = MTLVertexStepFunctionPerVertex;

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
            id<MTLFunction> vertexFunction = [myLibrary newFunctionWithName:@"pbr_vert_main"];
            id<MTLFunction> fragmentFunction = [myLibrary newFunctionWithName:@"pbr_frag_main"];

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
                NSLog(@"Failed to created render pipeline state, error %@", error);
            }
        }
        break;
        case PIPELINE_TYPE::COMPUTE:
        {
            id<MTLFunction> compFunction = [myLibrary newFunctionWithName:@"integrateBRDF_comp_main"];

            MTLComputePipelineDescriptor *pipelineStateDescriptor = [[MTLComputePipelineDescriptor alloc] init];

            pipelineStateDescriptor.label = [NSString stringWithCString:pState->pipelineStateName.c_str()
                                                    encoding:[NSString defaultCStringEncoding]];
            pipelineStateDescriptor.computeFunction = compFunction;

            MTLRenderPipelineReflection* reflectionObj;
            MTLPipelineOption option = MTLPipelineOptionBufferTypeInfo | MTLPipelineOptionArgumentInfo;

            pState->mtlComputePipelineState =
                [_device newComputePipelineStateWithDescriptor:pipelineStateDescriptor options:option error:&error];
            if (!pState->mtlComputePipelineState)
            {
                NSLog(@"Failed to created compute pipeline state, error %@", error);
            }
        }
        break;
        default:
            assert(0);
    }

/*
    for (MTLArgument *arg in reflectionObj.vertexArguments)
    {
        NSLog(@"Found arg: %@\n", arg.name);

        if (arg.bufferDataType == MTLDataTypeStruct)
        {
            for( MTLStructMember* uniform in arg.bufferStructType.members )
            {
                NSLog(@"uniform: %@ type:%lu, location: %lu", uniform.name, (unsigned long)uniform.dataType, (unsigned long)uniform.offset);         
            }
        }
    }
*/

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
