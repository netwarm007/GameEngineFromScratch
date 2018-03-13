#pragma once
#include "OrbisMemoryManager.hpp"

namespace My {
	class OrbisOnionHeapAllocator 
	{
	public:
		OrbisOnionHeapAllocator();
		virtual ~OrbisOnionHeapAllocator();

		// allocate memory
		virtual void* allocate(size_t size, int alignment, const char* tag);
		// free memory
		virtual void release(void *pointer);
	};
}
