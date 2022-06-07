#include "StackAllocator.hpp"

using namespace My;

StackAllocator::StackAllocator(IMemoryManager* pMmgr, size_t page_size, size_t alignment) : IAllocator(pMmgr) {

}

void* StackAllocator::Allocate(size_t size) { return nullptr; }

void StackAllocator::Free(void* p) {}

void StackAllocator::FreeAll() {}
