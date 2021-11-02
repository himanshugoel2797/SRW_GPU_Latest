/*****************************************************************
* 
* CUDA Header file
*   
* Ruizi Li, May 2021
*
*****************************************************************/
   
#ifndef _SRRADMNPGPU_CUDA_H
#define _SRRADMNPGPU_CUDA_H

// CUDA runtime
//#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

#ifdef _OFFLOAD_GPU
__global__ void MutualIntensityComponentCUDA(float *mo, float *v1, float *v2, long nxy, double iter, long long tot_iter, int PloCom);
#endif

#endif
