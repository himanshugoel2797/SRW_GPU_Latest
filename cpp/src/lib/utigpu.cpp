/************************************************************************//**
 * File: utigpu.cpp
 * Description: Auxiliary utilities to support GPU management
 *
 * @author H.Goel
 ***************************************************************************/

#include "utigpu.h"

#ifdef _OFFLOAD_GPU
#include <cuda_runtime.h>
#endif

static bool isGPUAvailable = false;
static bool isGPUEnabled = false;
static bool GPUAvailabilityTested = false;

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

bool UtiGPU::GPUAvailable()
{
	CheckGPUAvailability();
	return isGPUAvailable;
}

bool UtiGPU::GPUEnabled() 
{
	return isGPUEnabled;
}

void UtiGPU::SetGPUStatus(bool enabled)
{
	isGPUEnabled = enabled && GPUAvailable();
}