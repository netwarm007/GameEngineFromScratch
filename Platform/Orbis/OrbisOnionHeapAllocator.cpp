#include "OrbisOnionHeapAllocator.hpp"

using namespace My;
using namespace std;

OrbisOnionHeapAllocator::OrbisOnionHeapAllocator()
{

}

OrbisOnionHeapAllocator::~OrbisOnionHeapAllocator()
{

}

void * OrbisOnionHeapAllocator::allocate(size_t size, int alignment, const char * tag)
{
	void* result = nullptr;

	const uint32_t mask = alignment - 1;
	m_topOnion = (m_topOnion + mask) & ~mask;
	result = m_allocationOnion[m_allocationsOnion++] = m_pOnionMemory + m_topOnion;
	m_topOnion += size;
	assert(m_topOnion <= static_cast<off_t>(m_nOnionMemorySize));

	return result;
}

void OrbisOnionHeapAllocator::release(void * pointer)
{
	if (m_allocationOnion > 0) {
		uint8_t* lastPointer = m_allocationOnion[m_allocationsOnion - 1];
		if (lastPointer == pointer) {
			m_topOnion = lastPointer - m_pOnionMemory;
			m_allocationsOnion--;
			return;
		}
	}
}
