#include "StackAllocator.hpp"

using namespace My;

StackAllocator::StackAllocator() = default;

StackAllocator::StackAllocator(size_t page_size, size_t alignment) {}

StackAllocator::~StackAllocator() = default;

void* StackAllocator::Allocate(size_t size) { return nullptr; }

void StackAllocator::Free(void* p) {}

void StackAllocator::FreeAll() {}
