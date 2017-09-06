#pragma once
#include "IRuntimeModule.hpp"
#include "Allocator.hpp"
#include <new>

namespace My {
    template<typename T, typename... Arguments>
    T* New(Arguments... parameters)
    {
        return new (Allocate(sizeof(T))) T(Arguments);
    }

    template<typename T>
    void Delete(T *p)
    {
        reinterpret_cast<T*>(p)->~T();
        Free(p, sizeof(T));
    }

	class MemoryManager : implements IRuntimeModule
	{
	public:
		virtual ~MemoryManager() {}

        virtual int Initialize();
        virtual void Finalize();
        virtual void Tick();

        void* Allocate(size_t size);
        void  Free(void* p, size_t size);
    private:
        static size_t*        m_pBlockSizeLookup;
        static Allocator*     m_pAllocators;
    private:
        static Allocator* LookUpAllocator(size_t size);
	};
}

