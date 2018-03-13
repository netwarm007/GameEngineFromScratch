#pragma once
#include <cstdint>
#include <cstdlib>
#include "Interface.hpp"

namespace My {
   class IAllocator {
        public:
                virtual ~IAllocator(); 

                virtual void* Allocate(size_t size) = 0;
                virtual void  Free(void* p) = 0;
                virtual void  FreeAll() = 0;
    };
}