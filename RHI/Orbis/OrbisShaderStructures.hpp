#pragma once

namespace My {
	/** @brief A simulated VGPR
	*/
	union Reg32
	{
		enum { kBias = 127 };
		struct
		{
			uint32_t m_mantissa : 23;
			uint32_t m_exponent : 8;
			uint32_t m_sign : 1;
		} bits;
		float    f;
		uint32_t u;
		int32_t  i;
	};

		struct VsShader
	{
		sce::Gnmx::VsShader *m_shader;
		void *m_fetchShader;
		sce::Gnmx::InputOffsetsCache m_offsetsTable;
	};

	struct PsShader
	{
		sce::Gnmx::PsShader *m_shader;
		sce::Gnmx::InputOffsetsCache m_offsetsTable;
	};

	struct CsShader
	{
		sce::Gnmx::CsShader *m_shader;
		sce::Gnmx::InputOffsetsCache m_offsetsTable;
	};

	struct HsShader
	{
		sce::Gnmx::HsShader *m_shader;
		sce::Gnmx::InputOffsetsCache m_offsetsTable;
	};

	struct EsShader
	{
		sce::Gnmx::EsShader *m_shader;
		void *m_fetchShader;
		sce::Gnmx::InputOffsetsCache m_offsetsTable;
	};

	struct LsShader
	{
		sce::Gnmx::LsShader *m_shader;
		void *m_fetchShader;
		sce::Gnmx::InputOffsetsCache m_offsetsTable;
	};

	struct GsShader
	{
		sce::Gnmx::GsShader *m_shader;
		sce::Gnmx::InputOffsetsCache m_offsetsTable;
	};

	struct CsVsShader
	{
		sce::Gnmx::CsVsShader *m_shader;
		void *m_fetchShaderVs;
		void *m_fetchShaderCs;
		sce::Gnmx::InputOffsetsCache m_offsetsTableVs;
		sce::Gnmx::InputOffsetsCache m_offsetsTableCs;
	};

	struct EmbeddedPsShader
	{
		const uint32_t *m_source;
		const char *m_name;
        sce::Gnmx::PsShader *m_shader;
        sce::Gnmx::InputOffsetsCache m_offsetsTable;

		void initialize();
	};

	struct EmbeddedCsShader
	{
		const uint32_t *m_source;
		const char *m_name;
        sce::Gnmx::CsShader *m_shader;
        sce::Gnmx::InputOffsetsCache m_offsetsTable;

		void initialize();
	};

	struct EmbeddedVsShader
	{
		const uint32_t *m_source;
		const char *m_name;
        sce::Gnmx::VsShader *m_shader;
		void *m_fetchShader;
        sce::Gnmx::InputOffsetsCache m_offsetsTable;

		void initialize();
	};

	struct EmbeddedEsShader
	{
		const uint32_t *m_source;
		const char *m_name;
        sce::Gnmx::EsShader *m_shader;
		void *m_fetchShader;
        sce::Gnmx::InputOffsetsCache m_offsetsTable;

		void initialize();
	};

	struct EmbeddedGsShader
	{
		const uint32_t *m_source;
		const char *m_gsName;
		const char *m_vsName;
        sce::Gnmx::GsShader *m_shader;
        sce::Gnmx::InputOffsetsCache m_offsetsTable;

		void initialize();
	};

	struct EmbeddedLsShader
	{
		const uint32_t *m_source;

		const char *m_name;
        sce::Gnmx::LsShader *m_shader;
		void *m_fetchShader;
        sce::Gnmx::InputOffsetsCache m_offsetsTable;

		void initialize();
	};

	struct EmbeddedHsShader
	{
		const uint32_t *m_source;
		const char *m_name;
        sce::Gnmx::HsShader *m_shader;
        sce::Gnmx::InputOffsetsCache m_offsetsTable;

		void initialize();
	};

	struct EmbeddedCsVsShader
	{
		const uint32_t *m_source;
		const char *m_csName;
		const char *m_vsName;
        sce::Gnmx::CsVsShader *m_shader;
		void *m_fetchShaderVs;
		void *m_fetchShaderCs;
        sce::Gnmx::InputOffsetsCache m_offsetsTableVs;
        sce::Gnmx::InputOffsetsCache m_offsetsTableCs;

		void initialize();
	};

	struct EmbeddedShaders
	{
		EmbeddedCsShader **m_embeddedCsShader;
		uint64_t           m_embeddedCsShaders;
		EmbeddedPsShader **m_embeddedPsShader;
		uint64_t           m_embeddedPsShaders;
		EmbeddedVsShader **m_embeddedVsShader;
		uint64_t           m_embeddedVsShaders;
		EmbeddedEsShader **m_embeddedEsShader;
		uint64_t           m_embeddedEsShaders;
		EmbeddedGsShader **m_embeddedGsShader;
		uint64_t           m_embeddedGsShaders;
		EmbeddedLsShader **m_embeddedLsShader;
		uint64_t           m_embeddedLsShaders;
		EmbeddedHsShader **m_embeddedHsShader;
		uint64_t           m_embeddedHsShaders;
		EmbeddedCsVsShader **m_embeddedCsVsShader;
		uint64_t           m_embeddedCsVsShaders;
		void initialize();
	};
}

