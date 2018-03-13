#include "OrbisGalicHeapAllocator.hpp"

using namespace My;
using namespace std;

OrbisGalicHeapAllocator::OrbisGalicHeapAllocator()
{

}

OrbisGalicHeapAllocator::~OrbisGalicHeapAllocator()
{

}

void * OrbisGalicHeapAllocator::allocate(size_t size, int alignment, const char * tag)
{
	void* result = nullptr;

	assert(m_allocationsGalic < kMaximumAllocations);
	const uint32_t mask = alignment - 1;
	m_topGalic = (m_topGalic + mask) & ~mask;
	result = m_allocationGalic[m_allocationsGalic++] = m_pGalicMemory + m_topGalic;
	m_topGalic += size;
	assert(m_topGalic <= static_cast<off_t>(m_nGalicMemorySize));

	return result;
}

void My::OrbisGalicHeapAllocator::release(void * pointer)
{
	if (m_allocationGalic > 0) {
		uint8_t* lastPointer = m_allocationGalic[m_allocationsGalic - 1];
		if (lastPointer == pointer) {
			m_topGalic = lastPointer - m_pGalicMemory;
			m_allocationsGalic--;
			return;
		}
	}
}
