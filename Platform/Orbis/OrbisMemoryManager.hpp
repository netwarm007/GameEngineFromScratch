#pragma once
#include "MemoryManager.hpp"

#include <gnm.h>

using namespace sce;

namespace My {
	class OrbisMemoryManager : public MemoryManager
	{
	public:
		OrbisMemoryManager();
		~OrbisMemoryManager();

		virtual int Initialize();
		virtual void Finalize();
		virtual void Tick();
	private:
		void printMemoryUsage();

	private:
		SceLibcMallocManagedSize mmsize;
		uint8_t *m_pOnionMemory;
		size_t m_nOnionMemorySize;
		uint8_t *m_pGalicMemory;
		size_t m_nGalicMemorySize;
		off_t m_offsetOnion;
		off_t m_offsetGalic;
		uint64_t m_nAllignment;

		enum { kMaximumAllocations = 8192 };
		uint8_t *m_allocationOnion[kMaximumAllocations];
		uint64_t m_allocationsOnion;
		off_t m_topOnion;
		uint8_t *m_allocationGalic[kMaximumAllocations];
		uint64_t m_allocationsGalic;
		off_t m_topGalic;

		bool m_isInitialized;

		friend class OrbisOnionHeapAllocator;
		friend class OrbisGalicHeapAllocator;
	};
}

