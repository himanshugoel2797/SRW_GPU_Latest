/************************************************************************//**
 * File: utigpu.h
 * Description: GPU offloading detection and control
 * Project: Synchrotron Radiation Workshop
 * First release: 2000
 *
 * Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
 * All Rights Reserved
 *
 * @author H. Goel
 * @version 1.0
 ***************************************************************************/

#ifndef __UTIGPU_H
#define __UTIGPU_H

#include <cstdlib>
#include <stdio.h>

#ifdef _OFFLOAD_GPU
#include <cuda_runtime.h>
#endif

 //*************************************************************************
class UtiDev
{
public:
	static void Init();
	static void Fini();
	static bool GPUAvailable(); //CheckGPUAvailable etc
	static bool GPUEnabled();
	static void SetGPUStatus(bool enabled);
	template<typename T> static inline void malloc(T** ptr, size_t sz) {
#ifdef _OFFLOAD_GPU
			auto err = cudaMallocManaged<T>(ptr, sz);
			if (err != cudaSuccess)
				printf("Allocation Failure\r\n");
#else
			*ptr = std::malloc(sz);
#endif
	}

	static inline void free(void* ptr) {
#ifdef _OFFLOAD_GPU
		cudaFree(ptr);
#else
		std::free(ptr);
#endif
	}
};

//*************************************************************************
#endif