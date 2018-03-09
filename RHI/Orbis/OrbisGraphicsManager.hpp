#pragma once
#include <gnmx.h>
#include <video_out.h>
#include "GraphicsManager.hpp"
#include "OrbisShaderStructures.hpp"
#include "geommath.hpp"
#include "Scene.hpp"

namespace My {
	class OrbisGraphicsManager : public GraphicsManager
	{
	public:
		virtual int Initialize();
		virtual void Finalize();
		virtual void Tick();

		const sce::Gnm::OwnerHandle GetOwnerHandle();

	protected:
		void clearDepthStencil(sce::Gnmx::GnmxGfxContext *gfxc, const sce::Gnm::DepthRenderTarget *depthTarget);
		void clearDepthTarget(sce::Gnmx::GnmxGfxContext *gfxc, const sce::Gnm::DepthRenderTarget *depthTarget, const float depthValue);
		void clearMemoryToUints(sce::Gnmx::GnmxGfxContext * gfxc, void * destination, uint32_t destUints, uint32_t * source, uint32_t srcUints);
		void clearTexture(sce::Gnmx::GnmxGfxContext *gfxc, const sce::Gnm::Texture *texture, uint32_t *source, uint32_t sourceUints);
		void clearTexture(sce::Gnmx::GnmxGfxContext *gfxc, const sce::Gnm::Texture *texture, const Vector4f &color);
		void clearRenderTarget(sce::Gnmx::GnmxGfxContext *gfxc, const sce::Gnm::RenderTarget *renderTarget, uint32_t *source, uint32_t sourceUints);
		void clearRenderTarget(sce::Gnmx::GnmxGfxContext *gfxc, const sce::Gnm::RenderTarget *renderTarget, const Vector4f &color);
		
		void dataFormatEncoder(uint32_t * __restrict dest, uint32_t * __restrict destDwords, const Reg32 * __restrict src, const sce::Gnm::DataFormat dataFormat);
		void dataFormatDecoder(Reg32 * __restrict dest, const uint32_t * __restrict src, const sce::Gnm::DataFormat dataFormat);

		void synchronizeComputeToGraphics(sce::Gnmx::GnmxDrawCommandBuffer * dcb);

		void loadVsShaderFromMemory(const uint32_t *buff, const char *name, EmbeddedVsShader& shader);
		void loadPsShaderFromMemory(const uint32_t *buff, const char *name, EmbeddedPsShader& shader);
		void loadCsShaderFromMemory(const uint32_t *buff, const char *name, EmbeddedCsShader& shader);
		void loadVsShaderFromFile(const char *file_name, const char *name, VsShader& shader);
		void loadPsShaderFromFile(const char *file_name, const char *name, PsShader& shader);
		void loadCsShaderFromFile(const char *file_name, const char *name, CsShader& shader);

		int loadTextureFromGnf(const char *filename, const char *name, uint8_t textureIndex, sce::Gnm::Texture& texture);

		void setMeshVertexBufferFormat(sce::Gnm::Buffer* buffer, SimpleMesh& destMesh, const VertexElements* element, uint32_t elements);

		void registerRenderTargetForDisplay(sce::Gnm::RenderTarget *renderTarget);
		void requestFlip();
		void requestFlipAndWait();

	protected:
		enum { kSmallShaderThreshold = 512 * 1024 };
		uint8_t m_shaderLoadSmallBuffer[kSmallShaderThreshold]; // 512k buffer for fast shader loading

		// Video Out Infomation
		struct VideoInfo {
			volatile uint64_t* label;
			uint32_t label_num;
			int32_t flip_index;
			int32_t buffer_num;
			int32_t handle;
			SceKernelEqueue eq;
		};

		// Per-frame context
		struct Frame {
			sce::Gnmx::GnmxGfxContext commandBuffer;
			Constants *constants;
		};

		enum { kNumFB = 3 };
		static const sce::Gnm::ZFormat kZFormat = sce::Gnm::kZFormat32Float;
		static const sce::Gnm::StencilFormat kStencilFormat = sce::Gnm::kStencil8;
		static const bool kHtileEnabled = true;
		static const Vector4f kClearColor;
		
		enum { kTextureMaxCount = 100 };
		sce::Gnm::Texture m_texture[kTextureMaxCount];
		sce::Gnm::Sampler m_sampler[kTextureMaxCount];

		uint32_t m_targetWidth;
		uint32_t m_targetHeight;
		sce::Gnm::RenderTarget m_fbTarget[kNumFB];
		sce::Gnm::DepthRenderTarget m_depthTarget;

		// Shaders
		enum { kUserShaderMaxCount = 100 };
		VsShader m_vertexShader[kUserShaderMaxCount];
		PsShader m_pixelShader[kUserShaderMaxCount];
		CsShader m_computeShader[kUserShaderMaxCount];

		sce::Gnm::OwnerHandle m_owner;
		VideoInfo m_videoInfo;
		Frame m_frames[kNumFB];

		enum { kMeshMaxCount = 100 };
		SimpleMesh m_meshs[kMeshMaxCount];
		sce::Gnm::Buffer m_vertexBuffers[kMeshMaxCount][VertexElements::kVertexElemCount];

		OrbisGalicHeapAllocator m_galicHeapAllocator;
		OrbisOnionHeapAllocator m_onionHeapAllocator;
	};
}

