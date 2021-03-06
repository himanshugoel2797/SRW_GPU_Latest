/************************************************************************//**
 * File: utigpu.cpp
 * Description: Auxiliary utilities to support GPU management
 *
 * @author H.Goel
 ***************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <new>

#ifdef _OFFLOAD_GPU
#include <cuda_runtime.h>
#endif

#include "utidev.h"

static bool isGPUAvailable = false;
static bool isGPUEnabled = false;
static bool GPUAvailabilityTested = false;
static bool deviceOffloadInitialized = false;

static void CheckGPUAvailability() 
{
#ifdef _OFFLOAD_GPU
	if (!GPUAvailabilityTested)
	{
		isGPUAvailable = false;
		GPUAvailabilityTested = true;
		int deviceCount = 0;
		if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
			return;

		if (deviceCount < 1)
			return;

		isGPUAvailable = true;
	}
#else
	isGPUAvailable = false;
	isGPUEnabled = false;
	GPUAvailabilityTested = true;
#endif
}

bool UtiDev::GPUAvailable()
{
	CheckGPUAvailability();
	return isGPUAvailable;
}

bool UtiDev::GPUEnabled(gpuUsageArg_t *arg) 
{
	if (arg == NULL)
		return isGPUEnabled;
	if (*arg == 1) return false;
	return isGPUEnabled;
}

void UtiDev::SetGPUStatus(bool enabled)
{
	isGPUEnabled = enabled && GPUAvailable();
}

void UtiDev::Init() {
	deviceOffloadInitialized = true;
#ifdef _OFFLOAD_GPU
	cudaDeviceSynchronize();
#endif
}

void UtiDev::Fini() {
#ifdef _OFFLOAD_GPU
	cudaDeviceSynchronize();
#endif
	//deviceOffloadInitialized = false;
}


void* operator new[](std::size_t sz) // no inline, required by [replacement.functions]/3
{
	if (sz == 0)
		++sz; // avoid std::malloc(0) which may return nullptr on success

	void* ptr;
	if (deviceOffloadInitialized)
		UtiDev::malloc(&ptr, sz);
	else
		ptr = std::malloc(sz);
	if (ptr)
		return ptr;

	throw std::bad_alloc{}; // required by [new.delete.single]/3
}
void operator delete[](void* ptr) noexcept
{
	if (deviceOffloadInitialized)
		UtiDev::free(ptr);
	else
		std::free(ptr);
}