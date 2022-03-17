/****************************************
*
* Perform vector outer-products and additions in CUDA
*
* Ruizi Li, Apr 2021
*
*****************************************/
#ifdef _OFFLOAD_GPU

// Headers reference cuda/cuda-samples/0_simple
// System includes
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include <math.h>
#include "srradmnp.h"
#include "gmmeth.h"

#define CUDA_BLOCK_SIZE_MAX 16
#define CUDA_GRID_SIZE_MAX 16


typedef void(*stokesFun)(float*, float*, float*, float*, float*, double);

// mo = v1' v1 + v2' v2
// nxy: vector size in complex
template <int BLOCK_SIZE, stokesFun fun> __global__ void addVecOProdCUDA(float *mo, float *v1, float *v2, long nxy, double iter, long long tot_iter){
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;
	// Block dimension
	int Nx = blockDim.x;
	int Ny = blockDim.y;
	long vbsize = nxy/Nx;
	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	long bId = (Nx*by+bx);

	long offset_y = (long)((-1.+sqrt(1.+8.*bId))/2.0);
	if(bId==0) offset_y=0;
	long offset_x = bId - (long)((offset_y*(offset_y+1)+1)/2);
	while(offset_x < 0) { offset_y -= 1; offset_x = bId - (long)((offset_y*(offset_y+1)+1)/2); }
	while(offset_x > offset_y) { offset_y += 1; offset_x = bId - (long)((offset_y*(offset_y+1)+1)/2); }
	if(offset_y >= Ny || offset_y < 0) return;
	if(offset_x > offset_y || offset_x < 0) return;

	float *vb10 = v1 + 2*offset_x*vbsize;
	float *vb1p0 = v1 + 2*offset_y*vbsize;
	float *vb20 = v2 + 2*offset_x*vbsize;
	float *vb2p0 = v2 + 2*offset_y*vbsize;

	float *mb = mo + vbsize*((offset_y*vbsize+1)*offset_y+offset_x*2);

	__shared__ float V1p[2*BLOCK_SIZE], V2p[2*BLOCK_SIZE], V1[2*BLOCK_SIZE], V2[2*BLOCK_SIZE];

	for(long long n=0; n<tot_iter; n++)
	{
	float *vb1 = vb10 + 2*n*nxy;
	float *vb1p = vb1p0 + 2*n*nxy;
	float *vb2 = vb20 + 2*n*nxy;
	float *vb2p = vb2p0 + 2*n*nxy;
	for(long i=0; i<vbsize; i+=BLOCK_SIZE)
	{
	if(true){ //tx==0
		V1p[2*ty] = vb1p[(i+ty)*2]; V1p[2*ty+1] = -vb1p[(i+ty)*2+1];
		V2p[2*ty] = vb2p[(i+ty)*2]; V2p[2*ty+1] = -vb2p[(i+ty)*2+1];
	}
	long jsize = vbsize;
	if(offset_x==offset_y) jsize=i+1;
	for(long j=0; j<jsize; j+=BLOCK_SIZE)
	{
		float *MB = 0x0;
		if((offset_x<offset_y) || (j+tx<=i+ty))
		{
		MB = mb + (2*offset_y*vbsize+i+ty+1)*(i+ty) + (j+tx)*2;
		if(true){//ty==0
			V1[2*tx] = vb1[(j+tx)*2]; V1[2*tx+1] = vb1[(j+tx)*2+1];
			V2[2*tx] = vb2[(j+tx)*2]; V2[2*tx+1] = vb2[(j+tx)*2+1];
		}
		}
		__syncthreads();

		if((offset_x<offset_y) || (j+tx<=i+ty))
		{
		(*fun)(MB, V1+2*tx, V1p+2*ty, V2+2*tx, V2p+2*ty, iter);
#if 0
		MB[0] = (MB[0]*iter + V1p[2*ty]*V1[2*tx]-V1p[2*ty+1]*V1[2*tx+1] + V2p[2*ty]*V2[2*tx]-V2p[2*ty+1]*V2[2*tx+1])/(iter+1.);
		MB[1] = (MB[1]*iter + V1p[2*ty]*V1[2*tx+1]+V1p[2*ty+1]*V1[2*tx] + V2p[2*ty]*V2[2*tx+1]+V2p[2*ty+1]*V2[2*tx])/(iter+1.);
#endif
		}
		__syncthreads();
	}
	}
	__syncthreads();
	iter += 1.0;
	}
}


// s0
__global__ void StokesS0(float *mo, float *v1, float *v1p, float *v2, float *v2p, double iter){
	mo[0] = (mo[0]*iter + v1p[0]*v1[0]-v1p[1]*v1[1] + v2p[0]*v2[0]-v2p[1]*v2[1])/(iter+1.);
	mo[1] = (mo[1]*iter + v1p[0]*v1[1]+v1p[1]*v1[0] + v2p[0]*v2[1]+v2p[1]*v2[0])/(iter+1.);
}

// s1
__global__ void StokesS1(float *mo, float *v1, float *v1p, float *v2, float *v2p, double iter){
	mo[0] = (mo[0]*iter + v1p[0]*v1[0]-v1p[1]*v1[1] - v2p[0]*v2[0]+v2p[1]*v2[1])/(iter+1.);
	mo[1] = (mo[1]*iter + v1p[0]*v1[1]+v1p[1]*v1[0] - v2p[0]*v2[1]-v2p[1]*v2[0])/(iter+1.);
}

// Linear Horizontal Pol.
__global__ void StokesLH(float *mo, float *v1, float *v1p, float *v2, float *v2p, double iter){
	mo[0] = (mo[0]*iter + v1p[0]*v1[0]-v1p[1]*v1[1])/(iter+1.);
	mo[1] = (mo[1]*iter + v1p[0]*v1[1]+v1p[1]*v1[0])/(iter+1.);
}

// Linear Vertical Pol.
__global__ void StokesLV(float *mo, float *v1, float *v1p, float *v2, float *v2p, double iter){
	mo[0] = (mo[0]*iter + v2p[0]*v2[0]-v2p[1]*v2[1])/(iter+1.);
	mo[1] = (mo[1]*iter + v2p[0]*v2[1]+v2p[1]*v2[0])/(iter+1.);
}



void srTRadGenManip::MutualIntensityComponentCUDA(float *mo, float *v1, float *v2, long nxy, double iter, long long tot_iter, int PloCom){
	int block_size=CUDA_BLOCK_SIZE_MAX, grid_size;
	grid_size = (int)(nxy/block_size);
	if(grid_size*block_size!=nxy){
		printf("MutualIntensityComponentCUDA error: Unsupported Wavefront mesh size %d\nHave to be devidable by %d\nChange either the mesh size or CUDA_BLOCK_SIZE_MAX (make sure it's supported by the CUDA device) and restart\n", nxy, block_size);
		return;
	}
	if(grid_size>CUDA_GRID_SIZE_MAX) grid_size = CUDA_GRID_SIZE_MAX;
	dim3 threads(block_size, block_size);
	dim3 grid(grid_size, grid_size);
	if(block_size==16)
	switch(PloCom){
		case 0: // Linear H. Pol
			addVecOProdCUDA<16,&StokesLH><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
			break;
		case 1: // Linear V. Pol
			addVecOProdCUDA<16,&StokesLV><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
			break;
		case -1: // s0
			addVecOProdCUDA<16,&StokesS0><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
			break;
		case -2: // s1
			addVecOProdCUDA<16,&StokesS1><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
			break;
		default:
			addVecOProdCUDA<16,&StokesS0><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
	}
	if(block_size==8)
	switch(PloCom){
		case 0: // Linear H. Pol
			addVecOProdCUDA<8,&StokesLH><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
			break;
		case 1: // Linear V. Pol
			addVecOProdCUDA<8,&StokesLV><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
			break;
		case -1: // s0
			addVecOProdCUDA<8,&StokesS0><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
			break;
		case -2: // s1
			addVecOProdCUDA<8,&StokesS1><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
			break;
		default:
			addVecOProdCUDA<8,&StokesS0><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
	}
}

template <bool allStokesReq, bool intOverEnIsRequired>
__global__ void ExtractSingleElecIntensity2DvsXZ_Kernel(srTRadExtract RadExtract, srTSRWRadStructAccessData RadAccessData, srTRadGenManip *obj, double* arAuxInt, long long ie0, long long ie1, double InvStepRelArg, int Int_or_ReE)
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x); //nx range
    int iz = (blockIdx.y * blockDim.y + threadIdx.y); //nz range
    int iwfr = (blockIdx.z * blockDim.z + threadIdx.z); //nwfr range
    
	if (ix < RadAccessData.nx && iz < RadAccessData.nz && iwfr < RadAccessData.nWfr) 
    {
		int PolCom = RadExtract.PolarizCompon;
			
		//bool allStokesReq = (PolCom == -5); //OC18042020

		float* pI = 0, * pI1 = 0, * pI2 = 0, * pI3 = 0; //OC17042020
		double* pId = 0, * pI1d = 0, * pI2d = 0, * pI3d = 0;
		long ne = RadAccessData.ne, nx = RadAccessData.nx, nz = RadAccessData.nz, nwfr = RadAccessData.nWfr;
		//float *pI = 0;
		//DOUBLE *pId = 0;
		//double *pId = 0; //OC26112019 (related to SRW port to IGOR XOP8 on Mac)
		long long nxnz = ((long long)nx) * ((long long)nz);
		if (Int_or_ReE != 2)
		{
			pI = RadExtract.pExtractedData;
			if (allStokesReq) //OC17042020
			{
				pI1 = pI + nxnz; pI2 = pI1 + nxnz; pI3 = pI2 + nxnz;
			}
		}
		else
		{
			pId = RadExtract.pExtractedDataD;
			if (allStokesReq) //OC17042020
			{
				pI1d = pId + nxnz; pI2d = pI1d + nxnz; pI3d = pI2d + nxnz;
			}
		}

		float* pEx0 = RadAccessData.pBaseRadX;
		float* pEz0 = RadAccessData.pBaseRadZ;

		//long PerX = RadAccessData.ne << 1;
		//long PerZ = PerX*RadAccessData.nx;
		//long long PerX = RadAccessData.ne << 1;
		//long long PerZ = PerX*RadAccessData.nx;
		long long PerX = ((long long)ne) << 1; //OC18042020
		long long PerZ = PerX * nx;
		long long PerWfr = PerZ * nz;

		//bool intOverEnIsRequired = (RadExtract.Int_or_Phase == 7) && (ne > 1); //OC18042020
		double resInt, resInt1, resInt2, resInt3;
		double ConstPhotEnInteg = 1.;
		long long Two_ie0 = ie0 << 1, Two_ie1 = ie1 << 1; //OC26042019
		long ie;

		long offset = iwfr * PerWfr + iz * PerZ + ix * PerX;

		float* pEx_StartForX = pEx0 + offset;
		float* pEz_StartForX = pEz0 + offset;
		if (pI != 0) pI += offset / 2;
		if (pI1 != 0) pI1 += offset / 2;
		if (pI2 != 0) pI2 += offset / 2;
		if (pI3 != 0) pI3 += offset / 2;

		if (pId != 0) pId += offset / 2;
		if (pI1d != 0) pI1d += offset / 2;
		if (pI2d != 0) pI2d += offset / 2;
		if (pI3d != 0) pI3d += offset / 2;
		
		//long ixPerX = 0;

		float* pEx_St = pEx_StartForX + Two_ie0;
		float* pEz_St = pEz_StartForX + Two_ie0;
		float* pEx_Fi = pEx_StartForX + Two_ie1;
		float* pEz_Fi = pEz_StartForX + Two_ie1;

		if (intOverEnIsRequired) //OC140813
		{//integrate over photon energy / time
			double* tInt = arAuxInt;
			float* pEx_StAux = pEx_St;
			float* pEz_StAux = pEz_St;

			if (!allStokesReq) //OC17042020
			{
				for (ie = 0; ie < ne; ie++) //OC18042020
				//for(int ie=0; ie<RadAccessData.ne; ie++)
				{
					*(tInt++) = obj->IntensityComponent(pEx_StAux, pEz_StAux, PolCom, Int_or_ReE);
					pEx_StAux += 2;
					pEz_StAux += 2;
				}
				resInt = ConstPhotEnInteg * CGenMathMeth::Integ1D_FuncDefByArray(arAuxInt, ne, RadAccessData.eStep); //OC18042020
				//resInt = ConstPhotEnInteg*CGenMathMeth::Integ1D_FuncDefByArray(arAuxInt, RadAccessData.ne, RadAccessData.eStep);
			}
			else
			{
				for (ie = 0; ie < ne; ie++)
				{
					*(tInt++) = obj->IntensityComponent(pEx_StAux, pEz_StAux, -1, Int_or_ReE);
					pEx_StAux += 2; pEz_StAux += 2;
				}
				resInt = ConstPhotEnInteg * CGenMathMeth::Integ1D_FuncDefByArray(arAuxInt, ne, RadAccessData.eStep);

				tInt = arAuxInt; pEx_StAux = pEx_St; pEz_StAux = pEz_St;
				for (ie = 0; ie < ne; ie++)
				{
					*(tInt++) = obj->IntensityComponent(pEx_StAux, pEz_StAux, -2, Int_or_ReE);
					pEx_StAux += 2; pEz_StAux += 2;
				}
				resInt1 = ConstPhotEnInteg * CGenMathMeth::Integ1D_FuncDefByArray(arAuxInt, ne, RadAccessData.eStep);

				tInt = arAuxInt; pEx_StAux = pEx_St; pEz_StAux = pEz_St;
				for (ie = 0; ie < ne; ie++)
				{
					*(tInt++) = obj->IntensityComponent(pEx_StAux, pEz_StAux, -3, Int_or_ReE);
					pEx_StAux += 2; pEz_StAux += 2;
				}
				resInt2 = ConstPhotEnInteg * CGenMathMeth::Integ1D_FuncDefByArray(arAuxInt, ne, RadAccessData.eStep);

				tInt = arAuxInt; pEx_StAux = pEx_St; pEz_StAux = pEz_St;
				for (ie = 0; ie < ne; ie++)
				{
					*(tInt++) = obj->IntensityComponent(pEx_StAux, pEz_StAux, -4, Int_or_ReE);
					pEx_StAux += 2; pEz_StAux += 2;
				}
				resInt3 = ConstPhotEnInteg * CGenMathMeth::Integ1D_FuncDefByArray(arAuxInt, ne, RadAccessData.eStep);
			}
		}
		else
		{
			if (!allStokesReq) //OC18042020
			{
				resInt = obj->IntensityComponentSimpleInterpol(pEx_St, pEx_Fi, pEz_St, pEz_Fi, InvStepRelArg, PolCom, Int_or_ReE);
			}
			else //OC18042020
			{
				resInt = obj->IntensityComponentSimpleInterpol(pEx_St, pEx_Fi, pEz_St, pEz_Fi, InvStepRelArg, -1, Int_or_ReE);
				resInt1 = obj->IntensityComponentSimpleInterpol(pEx_St, pEx_Fi, pEz_St, pEz_Fi, InvStepRelArg, -2, Int_or_ReE);
				resInt2 = obj->IntensityComponentSimpleInterpol(pEx_St, pEx_Fi, pEz_St, pEz_Fi, InvStepRelArg, -3, Int_or_ReE);
				resInt3 = obj->IntensityComponentSimpleInterpol(pEx_St, pEx_Fi, pEz_St, pEz_Fi, InvStepRelArg, -4, Int_or_ReE);
			}
		}
		//OC140813
		if (pI != 0) *pI = (float)resInt;
		if (pId != 0) *pId = resInt; //OC18042020
		//if(pId != 0) *(pId++) = (double)resInt;
		if (allStokesReq) //OC18042020
		{
			if (RadExtract.pExtractedData != 0)
			{
				*pI1 = (float)resInt1; *pI2 = (float)resInt2; *pI3 = (float)resInt3;
			}
			else
			{
				*pI1d = resInt1; *pI2d = resInt2; *pI3d = resInt3;
			}
		}
	}
}

int srTRadGenManip::ExtractSingleElecIntensity2DvsXZParallel(srTRadExtract& RadExtract, double* arAuxInt, long long ie0, long long ie1, double InvStepRelArg)
{
	srTSRWRadStructAccessData& RadAccessData = *((srTSRWRadStructAccessData*)(hRadAccessData.ptr()));

    const int bs = 256;
    dim3 blocks(RadAccessData.nx / bs + ((RadAccessData.nx & (bs - 1)) != 0), RadAccessData.nz, RadAccessData.nWfr);
    dim3 threads(bs, 1);

	srTRadGenManip *local_copy;
	cudaMalloc((void**)&local_copy, sizeof(srTRadGenManip));
	cudaMemcpy(local_copy, this, sizeof(srTRadGenManip), cudaMemcpyHostToDevice);

	bool allStokesReq = (RadExtract.PolarizCompon == -5);
	bool intOverEnIsRequired = (RadExtract.Int_or_Phase == 7) && (RadAccessData.ne > 1);

	int Int_or_ReE = RadExtract.Int_or_Phase;
	if (Int_or_ReE == 7) Int_or_ReE = 0; //OC150813: time/phot. energy integrated single-e intensity requires "normal" intensity here

	if (allStokesReq)
		if (intOverEnIsRequired)
    		ExtractSingleElecIntensity2DvsXZ_Kernel<true, true> << <blocks, threads >> > (RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE);
		else
    		ExtractSingleElecIntensity2DvsXZ_Kernel<true, false> << <blocks, threads >> > (RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE);
	else
		if (intOverEnIsRequired)
    		ExtractSingleElecIntensity2DvsXZ_Kernel<false, true> << <blocks, threads >> > (RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE);
		else
    		ExtractSingleElecIntensity2DvsXZ_Kernel<false, false> << <blocks, threads >> > (RadExtract, RadAccessData, local_copy, arAuxInt, ie0, ie1, InvStepRelArg, Int_or_ReE);
	cudaFreeAsync(local_copy, 0);

#ifdef _DEBUG
	cudaStreamSynchronize(0);
	auto err = cudaGetLastError();
	printf("%s\r\n", cudaGetErrorString(err));
#endif
	return 0;
}
#endif