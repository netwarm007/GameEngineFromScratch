#include <assert.h>
#include <gnmx/shader_parser.h>
#include <sceconst.h>
#include "OrbisGraphicsManager.hpp"

using namespace sce;
using namespace My;

namespace My {
    const Vector4f OrbisGraphicsManager::kClearColor = Vector4f(0.0f, 0.0f, 0.0f, 1.0f);
}

int OrbisGraphicsManager::Initialize()
{
	// Init Gnm Resource Owner
	Gnm::registerOwner(&m_owner, "OrbisGraphicsManager");

	Gnm::GpuMode gpuMode = Gnm::getGpuMode();

	if (gpuMode == Gnm::GpuMode::kGpuModeNeo) {
		m_targetWidth = 3840;
		m_targetHeight = 2160;
	}
	else {
		m_targetWidth = 1920;
		m_targetHeight = 1080;
	}

	// Loading VS shaders
	loadVsShaderFromFile("Shader-bin/vex_vv.sb", "vex_vv", m_vertexShader[0]);
	loadVsShaderFromFile("Shader-bin/shader_vv.sb", "shader_vv", m_vertexShader[1]);

	// Loading PS shaders
	loadPsShaderFromFile("Shader-bin/pix_p.sb", "pix_p", m_pixelShader[0]);
	loadPsShaderFromFile("Shader-bin/pix_clear_p.sb", "pix_clear_p", m_pixelShader[1]);
	loadPsShaderFromFile("Shader-bin/shader_p.sb", "shader_p", m_pixelShader[2]);

	// Loading CS shaders
	loadCsShaderFromFile("Shader-bin/cs_set_uint_c.sb", "cs_set_uint_c", m_computeShader[0]);
	loadCsShaderFromFile("Shader-bin/cs_set_uint_fast_c.sb", "cs_set_uint_fast_c", m_computeShader[1]);

	//////////////////////////////////////////////
	// Create a synchronization point:
    //
	// Allocate a buffer in video shared memory for synchronization
	// Note: the pointer is "volatile" to make sure the compiler doesn't optimized out the read to that memory address.
	m_videoInfo.label_num = 1;
	m_videoInfo.label = static_cast<volatile uint64_t*>(m_onionHeapAllocator.allocate(sizeof(uint64_t), sizeof(uint64_t), MM_HINT::MEM_USAGE_SYNC_OBJECT, "m_videoInfo.label"));

	//////////////////////////////////////////////
	// Setup the Gfx context for each frame
	//
	for (int i = 0; i < kNumFB; i++)
	{
		const uint32_t kCommandBufferSizeInBytes = 2 * 1024 * 1024;
		const uint32_t kConstantBufferSizeInBytes = 256 * 1024;
		const uint32_t kNumRingEntries = 64;
		const uint32_t cueHeapSize = Gnmx::ConstantUpdateEngine::computeHeapSize(kNumRingEntries);
		void *constantUpdateEngine = m_galicHeapAllocator.allocate(cueHeapSize, Gnm::kAlignmentOfBufferInBytes, MM_HINT::MEM_USAGE_GENERAL, "CUE buffer");
		void *drawCommandBuffer = m_onionHeapAllocator.allocate(kCommandBufferSizeInBytes, Gnm::kAlignmentOfBufferInBytes, MM_HINT::MEM_USAGE_COMMAND, "Draw Command buffer");
		void *constantCommandBuffer = m_onionHeapAllocator.allocate(kConstantBufferSizeInBytes, Gnm::kAlignmentOfBufferInBytes, MM_HINT::MEM_USAGE_COMMAND, "Constant Update Command buffer");
		m_frames[i].commandBuffer.init(constantUpdateEngine, kNumRingEntries, drawCommandBuffer, kCommandBufferSizeInBytes, constantCommandBuffer, kCommandBufferSizeInBytes);
		m_frames[i].constants = static_cast<Constants *>(m_onionHeapAllocator.allocate(sizeof(m_frames[i].constants), Gnm::kAlignmentOfBufferInBytes, MM_HINT::MEM_USAGE_SHADER_PARAMETER, "Constant buffer"));
	}
	
    ///////////////////////////////////////////////
	// Setup the render target
    //
	{
		const Gnm::DataFormat format = Gnm::kDataFormatB8G8R8A8UnormSrgb;

		// Creating render target descriptor and initializing it.
		Gnm::TileMode tileMode;
		GpuAddress::computeSurfaceTileMode(gpuMode, &tileMode, GpuAddress::kSurfaceTypeColorTargetDisplayable, format, 1);

		sce::Gnm::RenderTargetSpec spec;
		spec.init();
		spec.m_width = m_targetWidth;
		spec.m_height = m_targetHeight;
		spec.m_pitch = 0;
		spec.m_numSlices = 1;
		spec.m_colorFormat = format;
		spec.m_colorTileModeHint = tileMode;
		spec.m_minGpuMode = gpuMode;
		spec.m_numSamples = Gnm::kNumSamples1;
		spec.m_numFragments = Gnm::kNumFragments1;
		spec.m_flags.enableCmaskFastClear = 0;
		spec.m_flags.enableFmaskCompression = 0;

		for (int i = 0; i < kNumFB; i++)
		{
			int32_t status = m_fbTarget[i].init(&spec);

			Gnm::SizeAlign fbSize = m_fbTarget[i].getColorSizeAlign();
			if (status != SCE_GNM_OK)
			{
				fbSize = sce::Gnm::SizeAlign(0, 0);
			}
			// Allocate render target buffer in video memeory.
			void* fbBaseAddr = m_galicHeapAllocator.allocate(fbSize.m_size, fbSize.m_align, MM_HINT::MEM_USAGE_RENDERTARGET, "Rendering Target");

			// In order to simplify the code, the simplet are using a memset to clear the render target.
			// This method should NOT be used once the GPU start using the memory.
			memset(fbBaseAddr, 0xFF, fbSize.m_size);

			// Set render target memory base address (gpu address)
			m_fbTarget[i].setAddresses(fbBaseAddr, 0, 0);
		}

		registerRenderTargetForDisplay(m_fbTarget);
	}

    /////////////////////////////////////////////
	// Setup the depth buffer 
    //
	{
		// Compute the tiling mode for the depth buffer
		Gnm::DataFormat depthFormat = Gnm::DataFormat::build(kZFormat);
		Gnm::TileMode depthTileMode;
		int ret;

		ret = GpuAddress::computeSurfaceTileMode(
			gpuMode, // NEO or Base
			&depthTileMode,									// Tile mode pointer
			GpuAddress::kSurfaceTypeDepthOnlyTarget,		// Surface type
			depthFormat,									// Surface format
			1);												// Elements per pixel
		SCE_GNM_ASSERT(ret == SCE_GNM_OK);

		// Initialize the depth buffer descriptor
		Gnm::SizeAlign stencilSizeAlign;
		Gnm::SizeAlign htileSizeAlign;

		Gnm::DepthRenderTargetSpec spec;
		spec.init();
		spec.m_width = m_targetWidth;
		spec.m_height = m_targetHeight;
		spec.m_pitch = 0;
		spec.m_numSlices = 1;
		spec.m_zFormat = depthFormat.getZFormat();
		spec.m_stencilFormat = kStencilFormat;
		spec.m_minGpuMode = gpuMode;
		spec.m_numFragments = Gnm::kNumFragments1;
		spec.m_flags.enableHtileAcceleration = kHtileEnabled ? 1 : 0;

		ret = m_depthTarget.init(&spec);
		SCE_GNM_ASSERT(ret == SCE_GNM_OK);

		Gnm::SizeAlign depthTargetSizeAlign = m_depthTarget.getZSizeAlign();

		// Initialize the HTILE buffer, if enabled
		if (kHtileEnabled)
		{
			htileSizeAlign = m_depthTarget.getHtileSizeAlign();
			void* htileMemory = m_galicHeapAllocator.allocate(htileSizeAlign.m_size, htileSizeAlign.m_align, MM_HINT::MEM_USAGE_RENDERTARGET, "HTile Buffer");
			SCE_GNM_ASSERT(htileMemory);

			m_depthTarget.setHtileAddress(htileMemory);
		}

		// Initialize the stencil buffer, if enabled
		void *stencilMemory = NULL;
		stencilSizeAlign = m_depthTarget.getStencilSizeAlign();
		if (kStencilFormat != Gnm::kStencilInvalid)
		{
			stencilMemory = m_galicHeapAllocator.allocate(stencilSizeAlign.m_size, stencilSizeAlign.m_align, MM_HINT::MEM_USAGE_RENDERTARGET, "Stencil Buffer");
			SCE_GNM_ASSERT(stencilMemory);
		}

		// Allocate the depth buffer
		void* depthMemory = m_galicHeapAllocator.allocate(depthTargetSizeAlign.m_size, depthTargetSizeAlign.m_align, MM_HINT::MEM_USAGE_RENDERTARGET, "Depth Buffer");
		SCE_GNM_ASSERT(depthMemory);
		m_depthTarget.setAddresses(depthMemory, stencilMemory);
	}

    /////////////////////////////////////////////////////
	// Setup the texture buffer
    //
	{
		// Initialize a Gnm::Texture object
		{

			Gnm::TextureSpec spec;
			spec.init();
			spec.m_textureType = Gnm::kTextureType2d;
			spec.m_width = 512;
			spec.m_height = 512;
			spec.m_depth = 1;
			spec.m_pitch = 0;
			spec.m_numMipLevels = 1;
			spec.m_numSlices = 1;
			spec.m_format = Gnm::kDataFormatR8G8B8A8UnormSrgb;
			spec.m_tileModeHint = Gnm::kTileModeDisplay_LinearAligned;
			spec.m_minGpuMode = gpuMode;
			spec.m_numFragments = Gnm::kNumFragments1;
			int32_t status = m_texture[0].init(&spec);

			if (status != SCE_GNM_OK)
				return status;
		}

		Gnm::SizeAlign textureSizeAlign = m_texture[0].getSizeAlign();
	
#if 0
		// Allocate the texture data using the alignment returned by initAs2d
		void *textureData = m_galicHeapAllocator.allocate(textureSizeAlign.m_size, textureSizeAlign.m_align, MM_HINT::MEM_USAGE_TEXTURE, "Texture Buffer");
		SCE_GNM_ASSERT(textureData);

		int ret;
		// Read the texture data
		AssetFilePtr fp = AssetLoaderOpenFile("Texture/texture.raw");
		ret = AssetFileRead(fp, textureSizeAlign.m_size, textureData);
		AssetLoaderCloseFile(fp);
		SCE_GNM_ASSERT(ret);

		// Set the base data address in the texture object
		m_texture[0].setBaseAddress(textureData);

		// Initialize the texture sampler
		m_sampler[0].init();
		m_sampler[0].setMipFilterMode(Gnm::kMipFilterModeNone);
		m_sampler[0].setXyFilterMode(Gnm::kFilterModeBilinear, Gnm::kFilterModeBilinear);

		ret = loadTextureFromGnf("Texture/icelogo-color.gnf", "icelogo-color", 0, m_texture[1]);
		SCE_GNM_ASSERT(ret == kGnfErrorNone);
		ret = loadTextureFromGnf("Texture/icelogo-normal.gnf", "icelogo-normal", 0, m_texture[2]);
		SCE_GNM_ASSERT(ret == kGnfErrorNone);

		for (int i = 0; i < 3; i++) {
			m_texture[i].setResourceMemoryType(Gnm::kResourceMemoryTypeRO);
		}
#endif
	}

    /////////////////////////////////////////////////////
	// Setup Vertex and Index buffer
    //
	{
		// Allocate the vertex buffer memory
		m_meshs[0].m_vertexCount = 4;
		m_meshs[0].m_vertexStride = sizeof(SimpleMeshVertex);
		m_meshs[0].m_primitiveType = PrimitiveType::kPrimitiveTypeTriList;
		m_meshs[0].m_vertexBufferSize = sizeof(kVertexData);
		m_meshs[0].m_vertexBuffer = static_cast<SimpleMeshVertex*>(m_galicHeapAllocator.allocate(
			m_meshs[0].m_vertexBufferSize, Gnm::kAlignmentOfBufferInBytes, MM_HINT::MEM_USAGE_VERTEX, "Vertex buffer"));
		SCE_GNM_ASSERT(m_meshs[0].m_vertexBuffer);

		// Allocate the index buffer memory
		m_meshs[0].m_indexCount = kIndexCount;
		m_meshs[0].m_indexType = IndexSize::kIndexSize16;
		m_meshs[0].m_indexBufferSize = sizeof(kIndexData);
		m_meshs[0].m_indexBuffer = static_cast<uint16_t*>(m_galicHeapAllocator.allocate(
			m_meshs[0].m_indexBufferSize, Gnm::kAlignmentOfBufferInBytes, MM_HINT::MEM_USAGE_INDEX, "Index buffer"));
		SCE_GNM_ASSERT(m_meshs[0].m_indexBuffer);

		// Copy the vertex/index data onto the GPU mapped memory
		memcpy(m_meshs[0].m_vertexBuffer, kVertexData, sizeof(kVertexData));
		memcpy(m_meshs[0].m_indexBuffer, kIndexData, sizeof(kIndexData));

		const VertexElements element[] = { VertexElements::kVertexPosition, VertexElements::kVertexColor, VertexElements::kVertexUv };
		const uint32_t elements = sizeof(element) / sizeof(element[0]);
		setMeshVertexBufferFormat(m_vertexBuffers[0], m_meshs[0], element, elements);

		MeshBuilder::BuildTorusMesh(m_onionHeapAllocator, 0.8f, 0.2f, 64, 32, 4, 1, m_meshs[1]);
		const VertexElements element1[] = { VertexElements::kVertexPosition, VertexElements::kVertexNormal, VertexElements::kVertexTangent, VertexElements::kVertexUv };
		const uint32_t elements1 = sizeof(element1) / sizeof(element1[0]);
		setMeshVertexBufferFormat(m_vertexBuffers[1], m_meshs[1], element1, elements1);
	}

    /////////////////////////////////////////////////////
	// Setup Constents
    //
    {
        m_viewToWorldMatrix = Matrix4::identity();
        Point3 lightPositionX = { -1.5, 4, 9 };
        Point3 lightTargetX = { 0, 0, 0 };
        Vector3 lightUpX = { 0.f, 1.f, 0.f };
        m_lightToWorldMatrix = inverse(Matrix4::lookAt(lightPositionX, lightTargetX, lightUpX));

        m_depthNear = 1.f;
        m_depthFar = 100.f;
        const float aspect = (float)m_targetWidth / (float)m_targetHeight;
        m_projectionMatrix = Matrix4::frustum(-aspect, aspect, -1, 1, m_depthNear, m_depthFar);
        Point3 eyePos = { 0,0,2.5f };
        Point3 lookAtPos = { 0, 0, 0 };
        Vector3 upVec = { 0, 1, 0 };
        SetViewToWorldMatrix(inverse(Matrix4::lookAt(eyePos, lookAtPos, upVec)));
    }

	return SCE_GNM_OK;
}

void OrbisGraphicsManager::Finalize()
{
}

void OrbisGraphicsManager::Tick()
{
	static float rotationAngle = 0.0f;

	// advance rotation
	rotationAngle += 0.25f*SCE_MATH_TWOPI / 60.0f;
	if (rotationAngle > SCE_MATH_TWOPI)
		rotationAngle -= SCE_MATH_TWOPI;

	////////////////////////////////////////////////////
	// Setup the Command Buffer:
	//

	// Create and initiallize a draw command buffer
	// Once created (and after each resetBuffer call), it is highly recommended to call: initializeDefaultHardwareState()
	int32_t index = m_videoInfo.flip_index;
	Frame *pFrame = &m_frames[index];
	Gnmx::GnmxGfxContext *gfxc = &pFrame->commandBuffer;
	
	// Reset the graphical context and initialize the hardware state
	gfxc->reset();
	gfxc->initializeDefaultHardwareState();

	gfxc->waitUntilSafeForRendering(m_videoInfo.handle, index);

	gfxc->setupScreenViewport(0, 0, m_targetWidth, m_targetHeight, 0.5f, 0.5f);

	// Unless using MRT, the pixel shader will output to render target 0.
	gfxc->setRenderTargetMask(0xF);
	gfxc->setActiveShaderStages(Gnm::kActiveShaderStagesVsPs);
	gfxc->setRenderTarget(0, &m_fbTarget[index]);
	gfxc->setDepthRenderTarget(&m_depthTarget);

	// Clear the color and the depth target
	clearRenderTarget(gfxc, &m_fbTarget[index], kClearColor);
	clearDepthTarget(gfxc, &m_depthTarget, 1.f);

	// Enable z-writes using a less comparison function
	Gnm::DepthStencilControl dsc;
	dsc.init();
	dsc.setDepthControl(Gnm::kDepthControlZWriteEnable, Gnm::kCompareFuncLess);
	dsc.setDepthEnable(true);
	gfxc->setDepthStencilControl(dsc);

	// Cull clock-wise backfaces
	Gnm::PrimitiveSetup primSetupReg;
	primSetupReg.init();
	primSetupReg.setCullFace(Gnm::kPrimitiveSetupCullFaceBack);
	primSetupReg.setFrontFace(Gnm::kPrimitiveSetupFrontFaceCcw);
	gfxc->setPrimitiveSetup(primSetupReg);

	// Setup an additive blending mode
	Gnm::BlendControl blendControl;
	blendControl.init();
	blendControl.setBlendEnable(true);
	blendControl.setColorEquation(
		Gnm::kBlendMultiplierSrcAlpha,
		Gnm::kBlendFuncAdd,
		Gnm::kBlendMultiplierOneMinusSrcAlpha);
	gfxc->setBlendControl(0, blendControl);

	// Set the vertex shader and the pixel shader
	gfxc->setActiveShaderStages(Gnm::kActiveShaderStagesVsPs);
	gfxc->setVsShader(m_vertexShader[0].m_shader, 0, m_vertexShader[0].m_fetchShader, &m_vertexShader[0].m_offsetsTable);
	gfxc->setPsShader(m_pixelShader[0].m_shader, &m_pixelShader[0].m_offsetsTable);

	// Draw the triangle.
	switch (m_meshs[1].m_primitiveType)
	{
	case PrimitiveType::kPrimitiveTypeLineList:
		gfxc->setPrimitiveType(Gnm::kPrimitiveTypeLineList);
		break;
	case PrimitiveType::kPrimitiveTypeLineListAdjacency:
		gfxc->setPrimitiveType(Gnm::kPrimitiveTypeLineListAdjacency);
		break;
	case PrimitiveType::kPrimitiveTypeLineLoop:
		gfxc->setPrimitiveType(Gnm::kPrimitiveTypeLineLoop);
		break;
	case PrimitiveType::kPrimitiveTypeLineStrip:
		gfxc->setPrimitiveType(Gnm::kPrimitiveTypeLineStrip);
		break;
	case PrimitiveType::kPrimitiveTypeLineStripAdjacency:
		gfxc->setPrimitiveType(Gnm::kPrimitiveTypeLineStripAdjacency);
		break;
	case PrimitiveType::kPrimitiveTypeNone:
		gfxc->setPrimitiveType(Gnm::kPrimitiveTypeNone);
		break;
	case PrimitiveType::kPrimitiveTypePatch:
		gfxc->setPrimitiveType(Gnm::kPrimitiveTypePatch);
		break;
	case PrimitiveType::kPrimitiveTypePointList:
		gfxc->setPrimitiveType(Gnm::kPrimitiveTypePointList);
		break;
	case PrimitiveType::kPrimitiveTypePolygon:
		gfxc->setPrimitiveType(Gnm::kPrimitiveTypePolygon);
		break;
	case PrimitiveType::kPrimitiveTypeQuadList:
		gfxc->setPrimitiveType(Gnm::kPrimitiveTypeQuadList);
		break;
	case PrimitiveType::kPrimitiveTypeQuadStrip:
		gfxc->setPrimitiveType(Gnm::kPrimitiveTypeQuadStrip);
		break;
	case PrimitiveType::kPrimitiveTypeRectList:
		gfxc->setPrimitiveType(Gnm::kPrimitiveTypeRectList);
		break;
	case PrimitiveType::kPrimitiveTypeTriFan:
		gfxc->setPrimitiveType(Gnm::kPrimitiveTypeTriFan);
		break;
	case PrimitiveType::kPrimitiveTypeTriList:
		gfxc->setPrimitiveType(Gnm::kPrimitiveTypeTriList);
		break;
	case PrimitiveType::kPrimitiveTypeTriListAdjacency:
		gfxc->setPrimitiveType(Gnm::kPrimitiveTypeTriListAdjacency);
		break;
	case PrimitiveType::kPrimitiveTypeTriStrip:
		gfxc->setPrimitiveType(Gnm::kPrimitiveTypeTriStrip);
		break;
	case PrimitiveType::kPrimitiveTypeTriStripAdjacency:
		gfxc->setPrimitiveType(Gnm::kPrimitiveTypeTriStripAdjacency);
		break;
	default:
		gfxc->setPrimitiveType(Gnm::kPrimitiveTypeNone);
		break;
	}
	
	// Setup the vertex buffer used by the ES stage (vertex shader)
	// Note that the setXxx methods of GfxContext which are used to set
	// shader resources (e.g.: V#, T#, S#, ...) map directly on the
	// Constants Update Engine. These methods do not directly produce PM4
	// packets in the command buffer. The CUE gathers all the resource
	// definitions and creates a set of PM4 packets later on in the
	// gfxc.drawXxx method.
	gfxc->setVertexBuffers(Gnm::kShaderStageVs, 0, m_meshs[1].m_vertexAttributeCount, m_vertexBuffers[1]);

	// Setup the texture and its sampler on the PS stage
	gfxc->setTextures(Gnm::kShaderStagePs, 0, 1, &m_texture[0]);
	gfxc->setSamplers(Gnm::kShaderStagePs, 0, 1, &m_sampler[0]);

	// Allocate the vertex shader constants from the command buffer
	Constants *constants = static_cast<Constants*>(
		gfxc->allocateFromCommandBuffer(sizeof(Constants), Gnm::kEmbeddedDataAlignment4));

	SCE_GNM_ASSERT(constants);

	// Update the constants
	{
		const Matrix4 m = Matrix4::rotationZYX(Vector3(rotationAngle, rotationAngle, 0.f));
		constants->m_modelView = transpose(m_worldToViewMatrix*m);
		constants->m_modelViewProjection = transpose(m_viewProjectionMatrix*m);
		constants->m_lightPosition = getLightPositionInViewSpace();
		constants->m_lightColor = getLightColor();
		constants->m_ambientColor = getAmbientColor();
		constants->m_lightAttenuation = Vector4f(1, 0, 0, 0);

		Gnm::Buffer constBuffer;
		constBuffer.initAsConstantBuffer(constants, sizeof(Constants));
		constBuffer.setResourceMemoryType(Gnm::kResourceMemoryTypeRO); // it's a constant buffer, so read-only is OK

		gfxc->setConstantBuffers(Gnm::kShaderStageVs, 0, 1, &constBuffer);
		gfxc->setConstantBuffers(Gnm::kShaderStagePs, 0, 1, &constBuffer);
	}

	switch (m_meshs[1].m_indexType)
	{
	case IndexSize::kIndexSize16:
		gfxc->setIndexSize(Gnm::kIndexSize16);
		break;
	case IndexSize::kIndexSize32:
		gfxc->setIndexSize(Gnm::kIndexSize32);
		break;
	default:
		SCE_GNM_ASSERT(false);
		break;
	}

	gfxc->drawIndex(m_meshs[0].m_indexCount, m_meshs[0].m_indexBuffer);

	// The following line will write at the address "label" the value 0x1.
	// This write will only occur once the previous commands have been completed.
	// It allows synchronization between CPU and GPU.
	//gfxc->writeAtEndOfPipe(Gnm::kEopFlushCbDbCaches, Gnm::kEventWriteDestMemory, (void *)m_label, Gnm::kEventWriteSource64BitsImmediate, 0x1, Gnm::kCacheActionNone, Gnm::kCachePolicyLru);
	gfxc->writeImmediateAtEndOfPipe(Gnm::kEopFlushCbDbCaches, (void *)m_videoInfo.label, m_videoInfo.label_num, Gnm::kCacheActionNone);

	*m_videoInfo.label = 2;

	void *dcbAddrGPU = gfxc->m_dcb.m_beginptr;
	// In order to submit a command buffer or dump the PM4 packet stream,
	// the code requires a start address and a size.
	const uint32_t cbSizeInByte = static_cast<uint32_t>(gfxc->m_dcb.m_cmdptr - gfxc->m_dcb.m_beginptr) * 4;
	uint32_t dcbSizeInBytes = cbSizeInByte;
	void *ccbAddrGPU = 0;
	uint32_t ccbSizeInBytes = 0;
	int32_t state = Gnm::submitCommandBuffers(1, &dcbAddrGPU, &dcbSizeInBytes, &ccbAddrGPU, &ccbSizeInBytes);
	SCE_GNM_ASSERT(state == sce::Gnm::kSubmissionSuccess);

	// Wait until it's done (waiting for the GPU to write "1" in "label")
	uint32_t wait = 0;
	while (*m_videoInfo.label != 1)
		++wait;

	// Display the render target on the screen
	requestFlipAndWait();

}

const Gnm::OwnerHandle OrbisGraphicsManager::GetOwnerHandle()
{
	return m_owner;
}

void OrbisGraphicsManager::clearDepthStencil(Gnmx::GnmxGfxContext * gfxc, const Gnm::DepthRenderTarget * depthTarget)
{
	gfxc->setRenderTargetMask(0x0);

	gfxc->setPsShader(m_pixelShader[1].m_shader, &m_pixelShader[1].m_offsetsTable);

	Vector4fUnaligned *constantBuffer = static_cast<Vector4fUnaligned*>(gfxc->allocateFromCommandBuffer(sizeof(Vector4fUnaligned), Gnm::kEmbeddedDataAlignment4));
	*constantBuffer = Vector4f(0.f, 0.f, 0.f, 0.f);
	Gnm::Buffer buffer;
	buffer.initAsConstantBuffer(constantBuffer, sizeof(Vector4fUnaligned));
	buffer.setResourceMemoryType(Gnm::kResourceMemoryTypeRO);
	gfxc->setConstantBuffers(Gnm::kShaderStagePs, 0, 1, &buffer);

	const uint32_t width = depthTarget->getWidth();
	const uint32_t height = depthTarget->getHeight();
	gfxc->setupScreenViewport(0, 0, width, height, 0.5f, 0.5f);
	const uint32_t firstSlice = depthTarget->getBaseArraySliceIndex();
	const uint32_t lastSlice = depthTarget->getLastArraySliceIndex();
	Gnm::DepthRenderTarget dtCopy = *depthTarget;
	for (uint32_t iSlice = firstSlice; iSlice <= lastSlice; ++iSlice)
	{
		dtCopy.setArrayView(iSlice, iSlice);
		gfxc->setDepthRenderTarget(&dtCopy);
		Gnmx::renderFullScreenQuad(gfxc);
	}

	gfxc->setRenderTargetMask(0xF);

	Gnm::DbRenderControl dbRenderControl;
	dbRenderControl.init();
	gfxc->setDbRenderControl(dbRenderControl);
}

void OrbisGraphicsManager::clearDepthTarget(Gnmx::GnmxGfxContext * gfxc, const Gnm::DepthRenderTarget * depthTarget, const float depthValue)
{
	gfxc->pushMarker("OrbisGraphicsManager::clearDepthTarget");

	Gnm::DbRenderControl dbRenderControl;
	dbRenderControl.init();
	dbRenderControl.setDepthClearEnable(true);
	gfxc->setDbRenderControl(dbRenderControl);

	Gnm::DepthStencilControl depthControl;
	depthControl.init();
	depthControl.setDepthControl(Gnm::kDepthControlZWriteEnable, Gnm::kCompareFuncAlways);
	depthControl.setStencilFunction(Gnm::kCompareFuncNever);
	depthControl.setDepthEnable(true);
	gfxc->setDepthStencilControl(depthControl);

	gfxc->setDepthClearValue(depthValue);

	clearDepthStencil(gfxc, depthTarget);

	gfxc->popMarker();
}

void OrbisGraphicsManager::clearMemoryToUints(Gnmx::GnmxGfxContext *gfxc, void *destination, uint32_t destUints, uint32_t *source, uint32_t srcUints)
{
	const bool srcUintsIsPowerOfTwo = (srcUints & (srcUints - 1)) == 0;

	CsShader& dispatchClearShader = m_computeShader[(srcUintsIsPowerOfTwo ? 1 : 0)];
	gfxc->setCsShader(dispatchClearShader.m_shader, &dispatchClearShader.m_offsetsTable);

	Gnm::Buffer destinationBuffer;
	destinationBuffer.initAsDataBuffer(destination, Gnm::kDataFormatR32Uint, destUints);
	destinationBuffer.setResourceMemoryType(Gnm::kResourceMemoryTypeGC);
	gfxc->setRwBuffers(Gnm::kShaderStageCs, 0, 1, &destinationBuffer);

	Gnm::Buffer sourceBuffer;
	sourceBuffer.initAsDataBuffer(source, Gnm::kDataFormatR32Uint, srcUints);
	sourceBuffer.setResourceMemoryType(Gnm::kResourceMemoryTypeRO);
	gfxc->setBuffers(Gnm::kShaderStageCs, 0, 1, &sourceBuffer);

	struct Constants
	{
		uint32_t m_destUints;
		uint32_t m_srcUints;
	};
	Constants *constants = (Constants*)gfxc->allocateFromCommandBuffer(sizeof(Constants), Gnm::kEmbeddedDataAlignment4);
	constants->m_destUints = destUints;
	constants->m_srcUints = srcUints - (srcUintsIsPowerOfTwo ? 1 : 0);
	Gnm::Buffer constantBuffer;
	constantBuffer.initAsConstantBuffer(constants, sizeof(*constants));
	gfxc->setConstantBuffers(Gnm::kShaderStageCs, 0, 1, &constantBuffer);

	gfxc->dispatch((destUints + Gnm::kThreadsPerWavefront - 1) / Gnm::kThreadsPerWavefront, 1, 1);

	synchronizeComputeToGraphics(&gfxc->m_dcb);
}

void OrbisGraphicsManager::clearTexture(Gnmx::GnmxGfxContext * gfxc, const Gnm::Texture * texture, uint32_t * source, uint32_t sourceUints)
{
	uint64_t totalTiledSize = 0;
	Gnm::AlignmentType totalTiledAlign;
	int32_t status = GpuAddress::computeTotalTiledTextureSize(&totalTiledSize, &totalTiledAlign, texture);
	SCE_GNM_ASSERT(status == GpuAddress::kStatusSuccess);
	clearMemoryToUints(gfxc, texture->getBaseAddress(), totalTiledSize / sizeof(uint32_t), source, sourceUints);
}

void OrbisGraphicsManager::clearTexture(Gnmx::GnmxGfxContext * gfxc, const Gnm::Texture * texture, const Vector4f & color)
{
	uint32_t *source = static_cast<uint32_t*>(gfxc->allocateFromCommandBuffer(sizeof(uint32_t) * 4, Gnm::kEmbeddedDataAlignment4));
	uint32_t dwords = 0;
	dataFormatEncoder(source, &dwords, (Reg32*)&color, texture->getDataFormat());
	clearTexture(gfxc, texture, source, dwords);
}

void OrbisGraphicsManager::clearRenderTarget(Gnmx::GnmxGfxContext * gfxc, const Gnm::RenderTarget * renderTarget, uint32_t * source, uint32_t sourceUints)
{
	// NOTE: this slice count is only valid if the array view hasn't changed since initialization!
	const uint32_t numSlices = renderTarget->getLastArraySliceIndex() - renderTarget->getBaseArraySliceIndex() + 1;
	clearMemoryToUints(gfxc, renderTarget->getBaseAddress(), renderTarget->getSliceSizeInBytes()*numSlices / sizeof(uint32_t), source, sourceUints);
}

void OrbisGraphicsManager::clearRenderTarget(Gnmx::GnmxGfxContext * gfxc, const Gnm::RenderTarget * renderTarget, const Vector4f & color)
{
	uint32_t *source = static_cast<uint32_t*>(gfxc->allocateFromCommandBuffer(sizeof(uint32_t) * 4, Gnm::kEmbeddedDataAlignment4));
	uint32_t dwords = 0;
	dataFormatEncoder(source, &dwords, (Reg32*)&color, renderTarget->getDataFormat());
	clearRenderTarget(gfxc, renderTarget, source, dwords);
}

void OrbisGraphicsManager::dataFormatEncoder(uint32_t *__restrict dest, uint32_t *__restrict destDwords, const Reg32 *__restrict src, const Gnm::DataFormat dataFormat)
{
	Gnm::SurfaceFormat surfaceFormat = dataFormat.getSurfaceFormat();
	SCE_GNM_ASSERT(surfaceFormat < sizeof(g_surfaceFormatInfo) / sizeof(g_surfaceFormatInfo[0]));
	const SurfaceFormatInfo *info = &g_surfaceFormatInfo[dataFormat.getSurfaceFormat()];
	SCE_GNM_ASSERT(info->m_format == surfaceFormat);

	info->m_encoder(info, dest, src, dataFormat);
	*destDwords = info->m_bitsPerElement <= 32 ? 1 : info->m_bitsPerElement / 32;
}

void OrbisGraphicsManager::dataFormatDecoder(Reg32 *__restrict dest, const uint32_t *__restrict src, const Gnm::DataFormat dataFormat)
{
	Gnm::SurfaceFormat surfaceFormat = dataFormat.getSurfaceFormat();
	SCE_GNM_ASSERT(surfaceFormat < sizeof(g_surfaceFormatInfo) / sizeof(g_surfaceFormatInfo[0]));
	const SurfaceFormatInfo *info = &g_surfaceFormatInfo[dataFormat.getSurfaceFormat()];
	SCE_GNM_ASSERT(info->m_format == surfaceFormat);

	info->m_decoder(info, dest, src, dataFormat);
}

void OrbisGraphicsManager::synchronizeComputeToGraphics(sce::Gnmx::GnmxDrawCommandBuffer *dcb)
{
	volatile uint64_t* label = (volatile uint64_t*)dcb->allocateFromCommandBuffer(sizeof(uint64_t), Gnm::kEmbeddedDataAlignment8); // allocate memory from the command buffer
	*label = 0x0; // set the memory to have the val 0
	dcb->writeAtEndOfShader(Gnm::kEosCsDone, const_cast<uint64_t*>(label), 0x1); // tell the CP to write a 1 into the memory only when all compute shaders have finished
	dcb->waitOnAddress(const_cast<uint64_t*>(label), 0xffffffff, Gnm::kWaitCompareFuncEqual, 0x1); // tell the CP to wait until the memory has the val 1
	dcb->flushShaderCachesAndWait(Gnm::kCacheActionWriteBackAndInvalidateL1andL2, 0, Gnm::kStallCommandBufferParserDisable); // tell the CP to flush the L1$ and L2$
}

void OrbisGraphicsManager::loadVsShaderFromMemory(const uint32_t * buff, const char *name, EmbeddedVsShader& shader)
{
	shader = { buff };
	shader.m_name = name;
	shader.initialize();
}

void OrbisGraphicsManager::loadPsShaderFromMemory(const uint32_t * buff, const char *name, EmbeddedPsShader& shader)
{
	shader = { buff };
	shader.m_name = name;
	shader.initialize();
}

void OrbisGraphicsManager::loadCsShaderFromMemory(const uint32_t * buff, const char *name, EmbeddedCsShader& shader)
{
	shader = { buff };
	shader.m_name = name;
	shader.initialize();
}

void OrbisGraphicsManager::loadVsShaderFromFile(const char * file_name, const char * name, VsShader & shader)
{
	AssetFilePtr fp = AssetLoaderOpenFile(file_name);
	int32_t size = AssetFileGetSize(fp);
	if (size <= kSmallShaderThreshold) {
		AssetLoaderRead(fp, m_shaderLoadSmallBuffer, size);
	}
	else {
		SCE_GNM_ASSERT(false);
	}

	EmbeddedVsShader tmp_shader = { reinterpret_cast<uint32_t*>(m_shaderLoadSmallBuffer) };
	loadVsShaderFromMemory(reinterpret_cast<uint32_t*>(m_shaderLoadSmallBuffer), name, tmp_shader);
	
	shader.m_shader = tmp_shader.m_shader;
	shader.m_fetchShader = tmp_shader.m_fetchShader;
	shader.m_offsetsTable = tmp_shader.m_offsetsTable;
}

void OrbisGraphicsManager::loadPsShaderFromFile(const char * file_name, const char * name, PsShader & shader)
{
	AssetFilePtr fp = AssetLoaderOpenFile(file_name);
	int32_t size = AssetFileGetSize(fp);
	if (size <= kSmallShaderThreshold) {
		AssetLoaderRead(fp, m_shaderLoadSmallBuffer, size);
	}
	else {
		SCE_GNM_ASSERT(false);
	}

	EmbeddedPsShader tmp_shader = { reinterpret_cast<uint32_t*>(m_shaderLoadSmallBuffer) };
	loadPsShaderFromMemory(reinterpret_cast<uint32_t*>(m_shaderLoadSmallBuffer), name, tmp_shader);

	shader.m_shader = tmp_shader.m_shader;
	shader.m_offsetsTable = tmp_shader.m_offsetsTable;
}

void OrbisGraphicsManager::loadCsShaderFromFile(const char * file_name, const char * name, CsShader & shader)
{
	AssetFilePtr fp = AssetLoaderOpenFile(file_name);
	int32_t size = AssetFileGetSize(fp);
	if (size <= kSmallShaderThreshold) {
		AssetLoaderRead(fp, m_shaderLoadSmallBuffer, size);
	}
	else {
		SCE_GNM_ASSERT(false);
	}

	EmbeddedCsShader tmp_shader = { reinterpret_cast<uint32_t*>(m_shaderLoadSmallBuffer) };
	loadCsShaderFromMemory(reinterpret_cast<uint32_t*>(m_shaderLoadSmallBuffer), name, tmp_shader);

	shader.m_shader = tmp_shader.m_shader;
	shader.m_offsetsTable = tmp_shader.m_offsetsTable;
}

int OrbisGraphicsManager::loadTextureFromGnf(const char * filename, const char * name, uint8_t textureIndex, Gnm::Texture& texture)
{
	int32_t length = 0;
	uint8_t* buf = AssetLoaderRead(filename, length);
	const Gnf::Header* header = reinterpret_cast<const Gnf::Header *>(buf);
	const Gnf::Contents* contents = reinterpret_cast<const Gnf::Contents *>(header + 1);

	if (header->m_magicNumber != sce::Gnf::kMagic)
		return kGnfErrorNotGnfFile;
	if (contents->m_alignment>31)
		return kGnfErrorAlignmentOutOfRange;
	if (contents->m_version == 1)
	{
		if ((contents->m_numTextures * sizeof(sce::Gnm::Texture) + sizeof(sce::Gnf::Contents)) != header->m_contentsSize)
			return kGnfErrorContentsSizeMismatch;
	}
	else
	{
		if (contents->m_version != sce::Gnf::kVersion)
			return kGnfErrorVersionMismatch;
		if (computeContentSize(contents) > header->m_contentsSize)
			return kGnfErrorContentsSizeMismatch;
	}

	const Gnm::SizeAlign pixelsSa = getTexturePixelsSize(contents, textureIndex);
	void* pixelsAddr = m_galicHeapAllocator.allocate(pixelsSa.m_size, pixelsSa.m_align, MM_HINT::MEM_USAGE_TEXTURE, "Texture buffer");
	if (!pixelsAddr) return kGnfErrorOutOfMemory;

	const void* pixelsSrc = buf + sizeof(*header) + header->m_contentsSize + getTexturePixelsByteOffset(contents, textureIndex);
	memcpy(pixelsAddr, pixelsSrc, pixelsSa.m_size);

	free(buf);

	texture = *patchTextures(const_cast<Gnf::Contents*>(contents), textureIndex, 1, &pixelsAddr);
	
	return kGnfErrorNone;
}

void OrbisGraphicsManager::setMeshVertexBufferFormat(Gnm::Buffer* buffer, SimpleMesh& destMesh, const VertexElements * element, uint32_t elements)
{
	destMesh.m_vertexAttributeCount = elements;
	while (elements--)
	{
		switch (*element++)
		{
		case VertexElements::kVertexPosition:
			buffer->initAsVertexBuffer(static_cast<uint8_t*>(destMesh.m_vertexBuffer) + offsetof(SimpleMeshVertex, m_position), Gnm::kDataFormatR32G32B32Float, sizeof(SimpleMeshVertex), destMesh.m_vertexCount);
			break;
		case VertexElements::kVertexColor:
			buffer->initAsVertexBuffer(static_cast<uint8_t*>(destMesh.m_vertexBuffer) + offsetof(SimpleMeshVertex, m_color), Gnm::kDataFormatR32G32B32Float, sizeof(SimpleMeshVertex), destMesh.m_vertexCount);
			break;
		case VertexElements::kVertexNormal:
			buffer->initAsVertexBuffer(static_cast<uint8_t*>(destMesh.m_vertexBuffer) + offsetof(SimpleMeshVertex, m_normal), Gnm::kDataFormatR32G32B32Float, sizeof(SimpleMeshVertex), destMesh.m_vertexCount);
			break;
		case VertexElements::kVertexTangent:
			buffer->initAsVertexBuffer(static_cast<uint8_t*>(destMesh.m_vertexBuffer) + offsetof(SimpleMeshVertex, m_tangent), Gnm::kDataFormatR32G32B32A32Float, sizeof(SimpleMeshVertex), destMesh.m_vertexCount);
			break;
		case VertexElements::kVertexUv:
			buffer->initAsVertexBuffer(static_cast<uint8_t*>(destMesh.m_vertexBuffer) + offsetof(SimpleMeshVertex, m_texture), Gnm::kDataFormatR32G32Float, sizeof(SimpleMeshVertex), destMesh.m_vertexCount);
			break;
		default:
			SCE_GNM_ASSERT(false);
			continue;
		}
		buffer->setResourceMemoryType(Gnm::kResourceMemoryTypeRO); // it's a vertex buffer, so read-only is OK
		++buffer;
	}
}

void OrbisGraphicsManager::registerRenderTargetForDisplay(Gnm::RenderTarget * renderTarget)
{
	const uint32_t kPlayerId = 0;
	int ret = sceVideoOutOpen(kPlayerId, SCE_VIDEO_OUT_BUS_TYPE_MAIN, 0, NULL);
	SCE_GNM_ASSERT_MSG(ret >= 0, "sceVideoOutOpen() returned error code %d.", ret);
	m_videoInfo.handle = ret;
	// Create Attribute
	SceVideoOutBufferAttribute attribute;
	sceVideoOutSetBufferAttribute(
		&attribute, 
		SCE_VIDEO_OUT_PIXEL_FORMAT_B8_G8_R8_A8_SRGB,
		SCE_VIDEO_OUT_TILING_MODE_TILE,
		SCE_VIDEO_OUT_ASPECT_RATIO_16_9,
		renderTarget->getWidth(), 
		renderTarget->getHeight(), 
		renderTarget->getWidth());

	ret = sceVideoOutSetFlipRate(m_videoInfo.handle, 0);
	SCE_GNM_ASSERT_MSG(ret >= 0, "sceVideoOutSetFlipRate() returned error code %d.", ret);
	// Prepare Equeue for Flip Sync
	ret = sceKernelCreateEqueue(&m_videoInfo.eq, __FUNCTION__);
	SCE_GNM_ASSERT_MSG(ret >= 0, "sceKernelCreateEqueue() returned error code %d.", ret);
	ret = sceVideoOutAddFlipEvent(m_videoInfo.eq, m_videoInfo.handle, NULL);
	SCE_GNM_ASSERT_MSG(ret >= 0, "sceVideoOutAddFlipEvent() returned error code %d.", ret);
	m_videoInfo.flip_index = 0;
	m_videoInfo.buffer_num = kNumFB;
	void* address[kNumFB];
	for (int32_t i = 0; i < kNumFB; i++) {
		address[i] = renderTarget[i].getBaseAddress();
	}
	ret = sceVideoOutRegisterBuffers(m_videoInfo.handle, 0, address, kNumFB, &attribute);
	SCE_GNM_ASSERT_MSG(ret >= 0, "sceVideoOutRegisterBuffers() returned error code %d.", ret);
}

void OrbisGraphicsManager::requestFlip()
{
	// Set Flip Request
	int ret = sceVideoOutSubmitFlip(m_videoInfo.handle, m_videoInfo.flip_index, SCE_VIDEO_OUT_FLIP_MODE_VSYNC, 0);
	SCE_GNM_ASSERT_MSG(ret >= 0, "sceVideoOutSubmitFlip() returned error code %d.", ret);
	m_videoInfo.flip_index = (m_videoInfo.flip_index + 1) % m_videoInfo.buffer_num;
	Gnm::submitDone();
}

void OrbisGraphicsManager::requestFlipAndWait()
{
	// Set Flip Request
	requestFlip();
	// Wait Flip
	SceKernelEvent ev;
	int ret, out;
	ret = sceKernelWaitEqueue(m_videoInfo.eq, &ev, 1, &out, 0);
}


void EmbeddedPsShader::initialize()
{
	Gnmx::ShaderInfo shaderInfo;
	Gnmx::parseShader(&shaderInfo, m_source);

	OrbisGalicHeapAllocator allocator;
	void *shaderBinary = allocator.allocate(shaderInfo.m_gpuShaderCodeSize, Gnm::kAlignmentOfShaderInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "PS Shader Binary");
	void *shaderHeader = allocator.allocate(shaderInfo.m_psShader->computeSize(), Gnm::kAlignmentOfBufferInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "PS Shader Header");

	memcpy(shaderBinary, shaderInfo.m_gpuShaderCode, shaderInfo.m_gpuShaderCodeSize);
	memcpy(shaderHeader, shaderInfo.m_psShader, shaderInfo.m_psShader->computeSize());

	m_shader = static_cast<Gnmx::PsShader*>(shaderHeader);
	m_shader->patchShaderGpuAddress(shaderBinary);

	if (0 != m_name && Gnm::kInvalidOwnerHandle != OrbisGraphicsManager::Instance().GetOwnerHandle())
	{
		Gnm::registerResource(nullptr, OrbisGraphicsManager::Instance().GetOwnerHandle(), m_shader->getBaseAddress(), shaderInfo.m_gpuShaderCodeSize, m_name, Gnm::kResourceTypeShaderBaseAddress, 0);
	}
	Gnmx::generateInputOffsetsCache(&m_offsetsTable, Gnm::kShaderStagePs, m_shader);
}

void EmbeddedCsShader::initialize()
{
	Gnmx::ShaderInfo shaderInfo;
	Gnmx::parseShader(&shaderInfo, m_source);

	OrbisGalicHeapAllocator allocator;
	void *shaderBinary = allocator.allocate(shaderInfo.m_gpuShaderCodeSize, Gnm::kAlignmentOfShaderInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "CS Shader Binary");
	void *shaderHeader = allocator.allocate(shaderInfo.m_psShader->computeSize(), Gnm::kAlignmentOfBufferInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "CS Shader Header");

	memcpy(shaderBinary, shaderInfo.m_gpuShaderCode, shaderInfo.m_gpuShaderCodeSize);
	memcpy(shaderHeader, shaderInfo.m_csShader, shaderInfo.m_csShader->computeSize());

	m_shader = static_cast<Gnmx::CsShader*>(shaderHeader);
	m_shader->patchShaderGpuAddress(shaderBinary);

	if (0 != m_name && Gnm::kInvalidOwnerHandle != OrbisGraphicsManager::Instance().GetOwnerHandle())
	{
		Gnm::registerResource(nullptr, OrbisGraphicsManager::Instance().GetOwnerHandle(), m_shader->getBaseAddress(), shaderInfo.m_gpuShaderCodeSize, m_name, Gnm::kResourceTypeShaderBaseAddress, 0);
	}
	Gnmx::generateInputOffsetsCache(&m_offsetsTable, Gnm::kShaderStageCs, m_shader);
}

void EmbeddedVsShader::initialize()
{
	Gnmx::ShaderInfo shaderInfo;
	Gnmx::parseShader(&shaderInfo, m_source);

	OrbisGalicHeapAllocator allocator;
	void *shaderBinary = allocator.allocate(shaderInfo.m_gpuShaderCodeSize, Gnm::kAlignmentOfShaderInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "VS Shader Binary");
	void *shaderHeader = allocator.allocate(shaderInfo.m_psShader->computeSize(), Gnm::kAlignmentOfBufferInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "VS Shader Header");

	memcpy(shaderBinary, shaderInfo.m_gpuShaderCode, shaderInfo.m_gpuShaderCodeSize);
	memcpy(shaderHeader, shaderInfo.m_vsShader, shaderInfo.m_vsShader->computeSize());

	m_shader = static_cast<Gnmx::VsShader*>(shaderHeader);
	m_shader->patchShaderGpuAddress(shaderBinary);

	// Allocate the memory for the fetch shader
	m_fetchShader = allocator.allocate(Gnmx::computeVsFetchShaderSize(m_shader), Gnm::kAlignmentOfFetchShaderInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "FS Shader");
	SCE_GNM_ASSERT(m_fetchShader);

	Gnm::FetchShaderInstancingMode *instancingData = NULL;
	uint32_t shaderModifier;
	Gnmx::generateVsFetchShader(m_fetchShader, &shaderModifier, m_shader, instancingData, instancingData != nullptr ? 256 : 0);

	if (0 != m_name && Gnm::kInvalidOwnerHandle != OrbisGraphicsManager::Instance().GetOwnerHandle())
	{
		Gnm::registerResource(nullptr, OrbisGraphicsManager::Instance().GetOwnerHandle(), m_shader->getBaseAddress(), shaderInfo.m_gpuShaderCodeSize, m_name, Gnm::kResourceTypeShaderBaseAddress, 0);
	}
	Gnmx::generateInputOffsetsCache(&m_offsetsTable, Gnm::kShaderStageVs, m_shader);
}

void EmbeddedEsShader::initialize()
{
	Gnmx::ShaderInfo shaderInfo;
	Gnmx::parseShader(&shaderInfo, m_source);

	OrbisGalicHeapAllocator allocator;
	void *shaderBinary = allocator.allocate(shaderInfo.m_gpuShaderCodeSize, Gnm::kAlignmentOfShaderInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "ES Shader Binary");
	void *shaderHeader = allocator.allocate(shaderInfo.m_psShader->computeSize(), Gnm::kAlignmentOfBufferInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "ES Shader Header");

	memcpy(shaderBinary, shaderInfo.m_gpuShaderCode, shaderInfo.m_gpuShaderCodeSize);
	memcpy(shaderHeader, shaderInfo.m_esShader, shaderInfo.m_esShader->computeSize());

	m_shader = static_cast<Gnmx::EsShader*>(shaderHeader);
	m_shader->patchShaderGpuAddress(shaderBinary);

	if (0 != m_name && Gnm::kInvalidOwnerHandle != OrbisGraphicsManager::Instance().GetOwnerHandle())
	{
		Gnm::registerResource(nullptr, OrbisGraphicsManager::Instance().GetOwnerHandle(), m_shader->getBaseAddress(), shaderInfo.m_gpuShaderCodeSize, m_name, Gnm::kResourceTypeShaderBaseAddress, 0);
	}
	Gnmx::generateInputOffsetsCache(&m_offsetsTable, Gnm::kShaderStageEs, m_shader);
}

void EmbeddedGsShader::initialize()
{
	Gnmx::ShaderInfo gsShaderInfo;
	Gnmx::ShaderInfo vsShaderInfo;
	Gnmx::parseGsShader(&gsShaderInfo, &vsShaderInfo, m_source);

	OrbisGalicHeapAllocator allocator;
	void *gsShaderBinary = allocator.allocate(gsShaderInfo.m_gpuShaderCodeSize, Gnm::kAlignmentOfShaderInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "GS Shader Binary");
	void *vsShaderBinary = allocator.allocate(vsShaderInfo.m_gpuShaderCodeSize, Gnm::kAlignmentOfShaderInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "VS Shader Binary");
	void *gsShaderHeader = allocator.allocate(gsShaderInfo.m_psShader->computeSize(), Gnm::kAlignmentOfBufferInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "GS Shader Header");

	memcpy(gsShaderBinary, gsShaderInfo.m_gpuShaderCode, gsShaderInfo.m_gpuShaderCodeSize);
	memcpy(vsShaderBinary, vsShaderInfo.m_gpuShaderCode, vsShaderInfo.m_gpuShaderCodeSize);
	memcpy(gsShaderHeader, gsShaderInfo.m_gsShader, gsShaderInfo.m_gsShader->computeSize());

	m_shader = static_cast<Gnmx::GsShader*>(gsShaderHeader);
	m_shader->patchShaderGpuAddresses(gsShaderBinary, vsShaderBinary);

	if (0 != m_gsName && Gnm::kInvalidOwnerHandle != OrbisGraphicsManager::Instance().GetOwnerHandle())
	{
		Gnm::registerResource(nullptr, OrbisGraphicsManager::Instance().GetOwnerHandle(), m_shader->getBaseAddress(), gsShaderInfo.m_gpuShaderCodeSize, m_gsName, Gnm::kResourceTypeShaderBaseAddress, 0);
	}
	if (0 != m_vsName && Gnm::kInvalidOwnerHandle != OrbisGraphicsManager::Instance().GetOwnerHandle())
	{
		Gnm::registerResource(nullptr, OrbisGraphicsManager::Instance().GetOwnerHandle(), m_shader->getCopyShader()->getBaseAddress(), vsShaderInfo.m_gpuShaderCodeSize, m_vsName, Gnm::kResourceTypeShaderBaseAddress, 0);
	}
	Gnmx::generateInputOffsetsCache(&m_offsetsTable, Gnm::kShaderStageGs, m_shader);
}

void EmbeddedLsShader::initialize()
{
	Gnmx::ShaderInfo shaderInfo;
	Gnmx::parseShader(&shaderInfo, m_source);

	OrbisGalicHeapAllocator allocator;
	void *shaderBinary = allocator.allocate(shaderInfo.m_gpuShaderCodeSize, Gnm::kAlignmentOfShaderInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "LS Shader Binary");
	void *shaderHeader = allocator.allocate(shaderInfo.m_psShader->computeSize(), Gnm::kAlignmentOfBufferInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "LS Shader Header");

	memcpy(shaderBinary, shaderInfo.m_gpuShaderCode, shaderInfo.m_gpuShaderCodeSize);
	memcpy(shaderHeader, shaderInfo.m_lsShader, shaderInfo.m_lsShader->computeSize());

	m_shader = static_cast<Gnmx::LsShader*>(shaderHeader);
	m_shader->patchShaderGpuAddress(shaderBinary);

	if (0 != m_name && Gnm::kInvalidOwnerHandle != OrbisGraphicsManager::Instance().GetOwnerHandle())
	{
		Gnm::registerResource(nullptr, OrbisGraphicsManager::Instance().GetOwnerHandle(), m_shader->getBaseAddress(), shaderInfo.m_gpuShaderCodeSize, m_name, Gnm::kResourceTypeShaderBaseAddress, 0);
	}
	Gnmx::generateInputOffsetsCache(&m_offsetsTable, Gnm::kShaderStageLs, m_shader);
}

void EmbeddedHsShader::initialize()
{
	Gnmx::ShaderInfo shaderInfo;
	Gnmx::parseShader(&shaderInfo, m_source);

	OrbisGalicHeapAllocator allocator;
	void *shaderBinary = allocator.allocate(shaderInfo.m_gpuShaderCodeSize, Gnm::kAlignmentOfShaderInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "HS Shader Binary");
	void *shaderHeader = allocator.allocate(shaderInfo.m_psShader->computeSize(), Gnm::kAlignmentOfBufferInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "HS Shader Header");

	memcpy(shaderBinary, shaderInfo.m_gpuShaderCode, shaderInfo.m_gpuShaderCodeSize);
	memcpy(shaderHeader, shaderInfo.m_hsShader, shaderInfo.m_hsShader->computeSize());

	m_shader = static_cast<Gnmx::HsShader*>(shaderHeader);
	m_shader->patchShaderGpuAddress(shaderBinary);

	if (0 != m_name && Gnm::kInvalidOwnerHandle != OrbisGraphicsManager::Instance().GetOwnerHandle())
	{
		Gnm::registerResource(nullptr, OrbisGraphicsManager::Instance().GetOwnerHandle(), m_shader->getBaseAddress(), shaderInfo.m_gpuShaderCodeSize, m_name, Gnm::kResourceTypeShaderBaseAddress, 0);
	}
	Gnmx::generateInputOffsetsCache(&m_offsetsTable, Gnm::kShaderStageHs, m_shader);
}

void EmbeddedCsVsShader::initialize()
{
	Gnmx::ShaderInfo csShaderInfo;
	Gnmx::ShaderInfo csvsShaderInfo;
	Gnmx::parseCsVsShader(&csvsShaderInfo, &csShaderInfo, m_source);

	OrbisGalicHeapAllocator allocator;
	void *vsShaderBinary = allocator.allocate(csvsShaderInfo.m_gpuShaderCodeSize, Gnm::kAlignmentOfShaderInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "CS Shader Binary");
	void *csShaderBinary = allocator.allocate(csShaderInfo.m_gpuShaderCodeSize, Gnm::kAlignmentOfShaderInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "CS Shader Binary");
	void *csvsShaderHeader = allocator.allocate(csShaderInfo.m_psShader->computeSize(), Gnm::kAlignmentOfBufferInBytes, MM_HINT::MEM_USAGE_SHADER_PROGRAM, "VS Shader Header");

	memcpy(vsShaderBinary, csvsShaderInfo.m_gpuShaderCode, csvsShaderInfo.m_gpuShaderCodeSize);
	memcpy(csShaderBinary, csShaderInfo.m_gpuShaderCode, csShaderInfo.m_gpuShaderCodeSize);
	memcpy(csvsShaderHeader, csvsShaderInfo.m_csvsShader, csvsShaderInfo.m_csvsShader->computeSize());

	m_shader = static_cast<Gnmx::CsVsShader*>(csvsShaderHeader);
	m_shader->patchShaderGpuAddresses(vsShaderBinary, csShaderBinary);

	if (0 != m_csName && Gnm::kInvalidOwnerHandle != OrbisGraphicsManager::Instance().GetOwnerHandle())
	{
		Gnm::registerResource(nullptr, OrbisGraphicsManager::Instance().GetOwnerHandle(), m_shader->getComputeShader()->getBaseAddress(), csShaderInfo.m_gpuShaderCodeSize, m_csName, Gnm::kResourceTypeShaderBaseAddress, 0);
	}
	if (0 != m_vsName && Gnm::kInvalidOwnerHandle != OrbisGraphicsManager::Instance().GetOwnerHandle())
	{
		Gnm::registerResource(nullptr, OrbisGraphicsManager::Instance().GetOwnerHandle(), m_shader->getVertexShader()->getBaseAddress(), csvsShaderInfo.m_gpuShaderCodeSize, m_vsName, Gnm::kResourceTypeShaderBaseAddress, 0);
	}
	generateInputOffsetsCacheForDispatchDraw(&m_offsetsTableCs, &m_offsetsTableVs, m_shader);
}

void EmbeddedShaders::initialize()
{
	for (uint32_t i = 0; i < m_embeddedCsShaders; ++i)
		m_embeddedCsShader[i]->initialize();
	for (uint32_t i = 0; i < m_embeddedPsShaders; ++i)
		m_embeddedPsShader[i]->initialize();
	for (uint32_t i = 0; i < m_embeddedVsShaders; ++i)
		m_embeddedVsShader[i]->initialize();
	for (uint32_t i = 0; i < m_embeddedEsShaders; ++i)
		m_embeddedEsShader[i]->initialize();
	for (uint32_t i = 0; i < m_embeddedGsShaders; ++i)
		m_embeddedGsShader[i]->initialize();
	for (uint32_t i = 0; i < m_embeddedLsShaders; ++i)
		m_embeddedLsShader[i]->initialize();
	for (uint32_t i = 0; i < m_embeddedHsShaders; ++i)
		m_embeddedHsShader[i]->initialize();
	for (uint32_t i = 0; i < m_embeddedCsVsShaders; ++i)
		m_embeddedCsVsShader[i]->initialize();
}

#endif
