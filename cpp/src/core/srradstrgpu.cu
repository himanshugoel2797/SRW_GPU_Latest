#ifdef _OFFLOAD_GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include <stdio.h>
#include <iostream>
#include <chrono>


__global__ void MultiplyElFieldByPhaseLin_Kernel(double xMult, double zMult, float* pBaseRadX, float* pBaseRadZ, int nWfr, int nz, int nx, int ne, float zStart, float zStep, float xStart, float xStep) {
    int iz = (blockIdx.x * blockDim.x + threadIdx.x); //nz range
    int ix = (blockIdx.y * blockDim.y + threadIdx.y); //nx range
    int iwfr = (blockIdx.z * blockDim.z + threadIdx.z); //nWfr range

    if (ix < nx && iz < nz && iwfr < nWfr) 
    {
		bool RadXisDefined = (pBaseRadX != 0);
		bool RadZisDefined = (pBaseRadZ != 0);

		double z = zStart + iz * zStep;
		double x = xStart + ix * xStep;
		double dPhZ = zMult * z;
		double dPh = dPhZ + xMult * x;
		double cosPh = cos(dPh), sinPh = sin(dPh);

		long long offset = iwfr * nz * nx * ne * 2 + iz * nx * ne * 2 + ix * ne * 2;
		float* tEx = pBaseRadX + offset;
		float* tEz = pBaseRadZ + offset;
		for (int ie = 0; ie < ne; ie++)
		{
			if (RadXisDefined)
			{
				//*(tEx++) *= a; *(tEx++) *= a;
				double newReEx = (*tEx) * cosPh - (*(tEx + 1)) * sinPh;
				double newImEx = (*tEx) * sinPh + (*(tEx + 1)) * cosPh;
				*(tEx++) = (float)newReEx; *(tEx++) = (float)newImEx;
			}
			if (RadZisDefined)
			{
				//*(tEz++) *= a; *(tEz++) *= a;
				double newReEz = (*tEz) * cosPh - (*(tEz + 1)) * sinPh;
				double newImEz = (*tEz) * sinPh + (*(tEz + 1)) * cosPh;
				*(tEz++) = (float)newReEz; *(tEz++) = (float)newImEz;
			}
		}
    }
}

void MultiplyElFieldByPhaseLin_CUDA(double xMult, double zMult, float* pBaseRadX, float* pBaseRadZ, int nWfr, int nz, int nx, int ne, float zStart, float zStep, float xStart, float xStep)
{
    const int bs = 256;
    dim3 blocks(nz / bs + ((nz & (bs - 1)) != 0), nx, nWfr);
    dim3 threads(bs, 1);
    MultiplyElFieldByPhaseLin_Kernel<< <blocks, threads >> > (xMult, zMult, pBaseRadX, pBaseRadZ, nWfr, nz, nx, ne, zStart, zStep, xStart, xStep);

#ifdef _DEBUG
    auto err = cudaGetLastError();
    printf("%s\r\n", cudaGetErrorString(err));
#endif
}

#endif