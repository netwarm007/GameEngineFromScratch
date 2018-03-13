#include <kernel.h>
#include <cstdio>
#include <cstdlib>
#include <mspace.h>
#include "OrbisMemoryManager.hpp"

using namespace My;
using namespace std;

size_t sceLibcHeapSize = 100 * 1024 * 1024;

OrbisMemoryManager::OrbisMemoryManager() 
	:  m_nOnionMemorySize(2UL * 1024 * 1024 * 1024), m_nGalicMemorySize(2UL * 1024 * 1024 * 1024), m_isInitialized(false)
{
}


OrbisMemoryManager::~OrbisMemoryManager()
{
}

int OrbisMemoryManager::Initialize()
{
	SCE_LIBC_INIT_MALLOC_MANAGED_SIZE(mmsize);

	if (!malloc_stats_fast(&mmsize)) {
		printf("Libc heap limit: %lu bytes\n", mmsize.maxSystemSize);
		printf("Libc heap maxium in use: %lu bytes\n", mmsize.maxInuseSize);
	}

	m_nAllignment = 2 * 1024 * 1024; // 2M

	// Allocate Onion Heap
	int retSys = sceKernelAllocateDirectMemory(0,
		SCE_KERNEL_MAIN_DMEM_SIZE,
		m_nOnionMemorySize,
		m_nAllignment,
		SCE_KERNEL_WB_ONION,
		&m_offsetOnion);

	assert(retSys == 0);

	m_pOnionMemory = nullptr;

	retSys = sceKernelMapDirectMemory(&reinterpret_cast<void*&>(m_pOnionMemory),
		m_nOnionMemorySize,
		SCE_KERNEL_PROT_CPU_READ | SCE_KERNEL_PROT_CPU_WRITE | SCE_KERNEL_PROT_GPU_ALL,
		0,
		m_offsetOnion,
		m_nAllignment);

	assert(retSys == 0);

	printf("Onion Heap start from 0x%012x(PHY) 0x%012x(VAS), size = %lu bytes, alignment = %lu bytes\n", 
		m_offsetOnion, m_pOnionMemory, m_nOnionMemorySize, m_nAllignment);

	// Allocate Galic Heap
	retSys = sceKernelAllocateDirectMemory(0,
		SCE_KERNEL_MAIN_DMEM_SIZE,
		m_nGalicMemorySize,
		m_nAllignment,
		SCE_KERNEL_WC_GARLIC,
		&m_offsetGalic);

	assert(retSys == 0);

	m_pGalicMemory = nullptr;

	retSys = sceKernelMapDirectMemory(&reinterpret_cast<void*&>(m_pGalicMemory),
		m_nGalicMemorySize,
		SCE_KERNEL_PROT_CPU_READ | SCE_KERNEL_PROT_CPU_WRITE | SCE_KERNEL_PROT_GPU_ALL,
		0,
		m_offsetGalic,
		m_nAllignment);

	assert(retSys == 0);

	printf("Galic Heap start from 0x%012x(PHY) 0x%012x(VAS), size = %lu bytes, alignment = %lu bytes\n", 
		m_offsetGalic, m_pGalicMemory, m_nGalicMemorySize, m_nAllignment);

	m_isInitialized = true;

	return 0;
}

void OrbisMemoryManager::Finalize()
{
	int retSys = sceKernelReleaseDirectMemory(m_offsetGalic, m_nGalicMemorySize);
	m_pGalicMemory = nullptr;
	m_offsetGalic = 0;
	assert(retSys == 0);

	retSys = sceKernelReleaseDirectMemory(m_offsetOnion, m_nOnionMemorySize);
	m_pOnionMemory = nullptr;
	m_offsetOnion = 0;
	assert(retSys == 0);

	m_isInitialized = false;
}

void OrbisMemoryManager::Tick()
{
}

void OrbisMemoryManager::printMemoryUsage()
{
	printf("Libc heap limit: %lu bytes\n", mmsize.maxSystemSize);
	printf("Libc heap maxium in use: %lu bytes\n", mmsize.maxInuseSize);
}


