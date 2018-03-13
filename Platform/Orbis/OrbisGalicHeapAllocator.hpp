#pragma once
#include "OrbisMemoryManager.hpp"

namespace My {
	class OrbisGalicHeapAllocator
	{
	public:
		OrbisGalicHeapAllocator();
		virtual ~OrbisGalicHeapAllocator();

		// allocate memory
		virtual void* allocate(size_t size, int alignment, const char* tag);
		// free memory
		virtual void release(void *pointer);
	};
}
