#pragma once
#include "cuda_runtime.h"
#include <sroptelm.h>
#include <srradstr.h>
#include <srstraux.h>

void TreatStronglyOscillatingTerm_CUDA(srTSRWRadStructAccessData& RadAccessData, bool TreatPolCompX, bool TreatPolCompZ, int ieStart, int ieBefEnd, double ConstRx, double ConstRz);
void MakeWfrEdgeCorrection_CUDA(srTSRWRadStructAccessData& RadAccessData, float* pDataEx, float* pDataEz, srTDataPtrsForWfrEdgeCorr& DataPtrs);

#ifdef __CUDACC__
template<class T> __global__ void RadPointModifierParallel_Kernel(srTSRWRadStructAccessData RadAccessData, void* pBufVars, T* tgt_obj)
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x); //nx range
	int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range

	if (ix < RadAccessData.nx && iz < RadAccessData.nz)
	{
		srTEFieldPtrs EPtrs;
		srTEXZ EXZ;
		EXZ.z = RadAccessData.zStart + iz * RadAccessData.zStep;
		EXZ.x = RadAccessData.xStart + ix * RadAccessData.xStep;

		for (int iwfr = 0; iwfr < RadAccessData.nWfr; iwfr++)
			for (int ie = 0; ie < RadAccessData.ne; ie++) {
				EXZ.e = RadAccessData.eStart + ie * RadAccessData.eStep;
				EXZ.aux_offset = RadAccessData.ne * RadAccessData.nx * RadAccessData.nz * 2 * iwfr + RadAccessData.ne * RadAccessData.nx * 2 * iz + RadAccessData.ne * 2 * ix + ie * 2;
				if (RadAccessData.pBaseRadX != 0)
				{
					EPtrs.pExRe = RadAccessData.pBaseRadX + EXZ.aux_offset;
					EPtrs.pExIm = EPtrs.pExRe + 1;
				}
				else
				{
					EPtrs.pExRe = 0;
					EPtrs.pExIm = 0;
				}
				if (RadAccessData.pBaseRadZ != 0)
				{
					EPtrs.pEzRe = RadAccessData.pBaseRadZ + EXZ.aux_offset;
					EPtrs.pEzIm = EPtrs.pEzRe + 1;
				}
				else
				{
					EPtrs.pEzRe = 0;
					EPtrs.pEzIm = 0;
				}

				tgt_obj->RadPointModifierPortable(EXZ, EPtrs, pBufVars);
			}
	}
}

template<class T> int RadPointModifierParallelImpl(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, T* tgt_obj)
{
	const int bs = 256;
	dim3 blocks(pRadAccessData->nx / bs + ((pRadAccessData->nx & (bs - 1)) != 0), pRadAccessData->nz);
	dim3 threads(bs, 1);

    printf("RadPointModifierParallelImpl on GPU\n");

    T* local_copy = NULL;
    cudaMallocManaged(&local_copy, sizeof(T));
    memcpy(local_copy, tgt_obj, sizeof(T));
	RadPointModifierParallel_Kernel<T> << <blocks, threads >> > (*pRadAccessData, pBufVars, local_copy);
    cudaDeviceSynchronize();
    cudaFree(local_copy);

#ifdef _DEBUG
	cudaStreamSynchronize(0);
	auto err = cudaGetLastError();
	printf("%s\r\n", cudaGetErrorString(err));
#endif

	return 0;
}
#endif