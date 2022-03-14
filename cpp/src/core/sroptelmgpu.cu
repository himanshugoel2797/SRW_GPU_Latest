#ifdef _OFFLOAD_GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include <stdio.h>
#include <iostream>
#include <chrono>
#include "sroptelm.h"
#include "sroptelmgpu.h"


__global__ void TreatStronglyOscillatingTerm_Kernel(srTSRWRadStructAccessData RadAccessData, bool TreatPolCompX, bool TreatPolCompZ, int ieStart, double ConstRx, double ConstRz) {
    int iz = (blockIdx.x * blockDim.x + threadIdx.x); //nz range
    int ix = (blockIdx.y * blockDim.y + threadIdx.y); //nx range
    int ie = (blockIdx.z * blockDim.z + threadIdx.z) + ieStart; //ne range
    
    if (ix < RadAccessData.nx && iz < RadAccessData.nz && ie < RadAccessData.ne + ieStart) 
    {
        double ePh = RadAccessData.eStart + RadAccessData.eStep * (ie - ieStart);
        if (RadAccessData.PresT == 1)
        {
            ePh = RadAccessData.avgPhotEn; //?? OC041108
        }

        double ConstRxE = ConstRx * ePh;
        double ConstRzE = ConstRz * ePh;
        if (RadAccessData.Pres == 1)
        {
            //double Lambda_m = 1.239854e-06/ePh;
            double Lambda_m = 1.239842e-06 / ePh;
            if (RadAccessData.PhotEnergyUnit == 1) Lambda_m *= 0.001; // if keV

            double Lambda_me2 = Lambda_m * Lambda_m;
            ConstRxE *= Lambda_me2;
            ConstRzE *= Lambda_me2;
        }

        double z = (RadAccessData.zStart - RadAccessData.zc) + (iz * RadAccessData.zStep);
        double PhaseAddZ = 0;
        if (RadAccessData.WfrQuadTermCanBeTreatedAtResizeZ) PhaseAddZ = ConstRzE * z * z;

        double x = (RadAccessData.xStart - RadAccessData.xc) + (ix * RadAccessData.xStep);
        double Phase = PhaseAddZ;
        if (RadAccessData.WfrQuadTermCanBeTreatedAtResizeX) Phase += ConstRxE * x * x;

        float SinPh, CosPh;
        sincosf(Phase, &SinPh, &CosPh);

        long long PerX = RadAccessData.ne << 1;
        long long PerZ = PerX * RadAccessData.nx;
        long long PerWfr = PerZ * RadAccessData.nz;
        long long offset = ie * 2 + iz * PerZ + ix * PerX;
        
        float* pEX_StartForWfr = RadAccessData.pBaseRadX + offset;
        float* pEZ_StartForWfr = RadAccessData.pBaseRadZ + offset;

        for (int iwfr = 0; iwfr < RadAccessData.nWfr; iwfr++)
        {
            long long iwfrPerWfr = iwfr * PerWfr;

            if (TreatPolCompX)
            {
                float* pExRe = pEX_StartForWfr + iwfrPerWfr;
                float* pExIm = pExRe + 1;
                double ExReNew = (*pExRe) * CosPh - (*pExIm) * SinPh;
                double ExImNew = (*pExRe) * SinPh + (*pExIm) * CosPh;
                *pExRe = (float)ExReNew; *pExIm = (float)ExImNew;
            }
            if (TreatPolCompZ)
            {
                float* pEzRe = pEZ_StartForWfr + iwfrPerWfr;
                float* pEzIm = pEzRe + 1;
                double EzReNew = (*pEzRe) * CosPh - (*pEzIm) * SinPh;
                double EzImNew = (*pEzRe) * SinPh + (*pEzIm) * CosPh;
                *pEzRe = (float)EzReNew; *pEzIm = (float)EzImNew;
            }
        }
    }
}

void TreatStronglyOscillatingTerm_CUDA(srTSRWRadStructAccessData& RadAccessData, bool TreatPolCompX, bool TreatPolCompZ, int ieStart, int ieBefEnd, double ConstRx, double ConstRz)
{
    const int bs = 256;
    dim3 blocks(RadAccessData.nz / bs + ((RadAccessData.nz & (bs - 1)) != 0), RadAccessData.nx, ieBefEnd - ieStart);
    dim3 threads(bs, 1);
    TreatStronglyOscillatingTerm_Kernel<< <blocks, threads >> > (RadAccessData, TreatPolCompX, TreatPolCompZ, ieStart, ConstRx, ConstRz);

#ifdef _DEBUG
    auto err = cudaGetLastError();
    printf("%s\r\n", cudaGetErrorString(err));
#endif
}

__global__ void MakeWfrEdgeCorrection_Kernel(srTSRWRadStructAccessData RadAccessData, float* pDataEx, float* pDataEz, srTDataPtrsForWfrEdgeCorr DataPtrs)
{
    int iz = (blockIdx.x * blockDim.x + threadIdx.x); //nz range
    int ix = (blockIdx.y * blockDim.y + threadIdx.y); //nx range
    int iwfr = (blockIdx.z * blockDim.z + threadIdx.z); //nwfr range

    if (ix < RadAccessData.nx && iz < RadAccessData.nz && iwfr < RadAccessData.nWfr)
    {
		double dxSt_dzSt = DataPtrs.dxSt * DataPtrs.dzSt;
		double dxSt_dzFi = DataPtrs.dxSt * DataPtrs.dzFi;
		double dxFi_dzSt = DataPtrs.dxFi * DataPtrs.dzSt;
		double dxFi_dzFi = DataPtrs.dxFi * DataPtrs.dzFi;

		long TwoNz = RadAccessData.nz << 1;
		long PerX = 2;
		long PerZ = PerX * RadAccessData.nx;
		long PerWfr = RadAccessData.nz * PerZ;

#define DATAPTR_CHECK(x, a) ((x >= 0) ? a[x + TwoNz * iwfr] : 0.)
		float fSSExRe = DATAPTR_CHECK(DataPtrs.fxStzSt_[0], DataPtrs.FFTArrXStEx);
		float fSSExIm = DATAPTR_CHECK(DataPtrs.fxStzSt_[1], DataPtrs.FFTArrXStEx);
		float fSSEzRe = DATAPTR_CHECK(DataPtrs.fxStzSt_[2], DataPtrs.FFTArrXStEz);
		float fSSEzIm = DATAPTR_CHECK(DataPtrs.fxStzSt_[3], DataPtrs.FFTArrXStEz);
		float fFSExRe = DATAPTR_CHECK(DataPtrs.fxFizSt_[0], DataPtrs.FFTArrXStEx);
		float fFSExIm = DATAPTR_CHECK(DataPtrs.fxFizSt_[1], DataPtrs.FFTArrXStEx);
		float fFSEzRe = DATAPTR_CHECK(DataPtrs.fxFizSt_[2], DataPtrs.FFTArrXStEz);
		float fFSEzIm = DATAPTR_CHECK(DataPtrs.fxFizSt_[3], DataPtrs.FFTArrXStEz);
		float fSFExRe = DATAPTR_CHECK(DataPtrs.fxStzFi_[0], DataPtrs.FFTArrXFiEx);
		float fSFExIm = DATAPTR_CHECK(DataPtrs.fxStzFi_[1], DataPtrs.FFTArrXFiEx);
		float fSFEzRe = DATAPTR_CHECK(DataPtrs.fxStzFi_[2], DataPtrs.FFTArrXFiEz);
		float fSFEzIm = DATAPTR_CHECK(DataPtrs.fxStzFi_[3], DataPtrs.FFTArrXFiEz);
		float fFFExRe = DATAPTR_CHECK(DataPtrs.fxFizFi_[0], DataPtrs.FFTArrXFiEx);
		float fFFExIm = DATAPTR_CHECK(DataPtrs.fxFizFi_[1], DataPtrs.FFTArrXFiEx);
		float fFFEzRe = DATAPTR_CHECK(DataPtrs.fxFizFi_[2], DataPtrs.FFTArrXFiEz);
		float fFFEzIm = DATAPTR_CHECK(DataPtrs.fxFizFi_[3], DataPtrs.FFTArrXFiEz);

		float bRe, bIm, cRe, cIm;

		long long Two_iz = iz << 1;
		long long Two_iz_p_1 = Two_iz + 1;
		long long Two_ix = ix << 1;
		long long Two_ix_p_1 = Two_ix + 1;

		float* tEx = pDataEx + iwfr * PerWfr + iz * PerZ + ix * PerX, * tEz = pDataEz + iwfr * PerWfr + iz * PerZ + ix * PerX;
		float ExRe = *tEx, ExIm = *(tEx + 1);
		float EzRe = *tEz, EzIm = *(tEz + 1);

		if (DataPtrs.dxSt != 0.)
		{
			float ExpXStRe = DataPtrs.ExpArrXSt[Two_ix], ExpXStIm = DataPtrs.ExpArrXSt[Two_ix_p_1];

			bRe = DataPtrs.FFTArrXStEx[Two_iz + TwoNz * iwfr]; bIm = DataPtrs.FFTArrXStEx[Two_iz_p_1 + TwoNz * iwfr];
			ExRe += (float)(DataPtrs.dxSt * (ExpXStRe * bRe - ExpXStIm * bIm));
			ExIm += (float)(DataPtrs.dxSt * (ExpXStRe * bIm + ExpXStIm * bRe));

			bRe = DataPtrs.FFTArrXStEz[Two_iz + TwoNz * iwfr]; bIm = DataPtrs.FFTArrXStEz[Two_iz_p_1 + TwoNz * iwfr];
			EzRe += (float)(DataPtrs.dxSt * (ExpXStRe * bRe - ExpXStIm * bIm));
			EzIm += (float)(DataPtrs.dxSt * (ExpXStRe * bIm + ExpXStIm * bRe));

			if (DataPtrs.dzSt != 0.)
			{
				bRe = DataPtrs.ExpArrZSt[Two_iz], bIm = DataPtrs.ExpArrZSt[Two_iz_p_1];
				cRe = ExpXStRe * bRe - ExpXStIm * bIm; cIm = ExpXStRe * bIm + ExpXStIm * bRe;

				ExRe += (float)(dxSt_dzSt * (fSSExRe * cRe - fSSExIm * cIm));
				ExIm += (float)(dxSt_dzSt * (fSSExRe * cIm + fSSExIm * cRe));
				EzRe += (float)(dxSt_dzSt * (fSSEzRe * cRe - fSSEzIm * cIm));
				EzIm += (float)(dxSt_dzSt * (fSSEzRe * cIm + fSSEzIm * cRe));
			}
			if (DataPtrs.dzFi != 0.)
			{
				bRe = DataPtrs.ExpArrZFi[Two_iz], bIm = DataPtrs.ExpArrZFi[Two_iz_p_1];
				cRe = ExpXStRe * bRe - ExpXStIm * bIm; cIm = ExpXStRe * bIm + ExpXStIm * bRe;

				ExRe -= (float)(dxSt_dzFi * (fSFExRe * cRe - fSFExIm * cIm));
				ExIm -= (float)(dxSt_dzFi * (fSFExRe * cIm + fSFExIm * cRe));
				EzRe -= (float)(dxSt_dzFi * (fSFEzRe * cRe - fSFEzIm * cIm));
				EzIm -= (float)(dxSt_dzFi * (fSFEzRe * cIm + fSFEzIm * cRe));
			}
		}
		if (DataPtrs.dxFi != 0.)
		{
			float ExpXFiRe = DataPtrs.ExpArrXFi[Two_ix], ExpXFiIm = DataPtrs.ExpArrXFi[Two_ix_p_1];

			bRe = DataPtrs.FFTArrXFiEx[Two_iz + TwoNz * iwfr]; bIm = DataPtrs.FFTArrXFiEx[Two_iz_p_1 + TwoNz * iwfr];
			ExRe -= (float)(DataPtrs.dxFi * (ExpXFiRe * bRe - ExpXFiIm * bIm));
			ExIm -= (float)(DataPtrs.dxFi * (ExpXFiRe * bIm + ExpXFiIm * bRe));

			bRe = DataPtrs.FFTArrXFiEz[Two_iz + TwoNz * iwfr]; bIm = DataPtrs.FFTArrXFiEz[Two_iz_p_1 + TwoNz * iwfr];
			EzRe -= (float)(DataPtrs.dxFi * (ExpXFiRe * bRe - ExpXFiIm * bIm));
			EzIm -= (float)(DataPtrs.dxFi * (ExpXFiRe * bIm + ExpXFiIm * bRe));

			if (DataPtrs.dzSt != 0.)
			{
				bRe = DataPtrs.ExpArrZSt[Two_iz], bIm = DataPtrs.ExpArrZSt[Two_iz_p_1];
				cRe = ExpXFiRe * bRe - ExpXFiIm * bIm; cIm = ExpXFiRe * bIm + ExpXFiIm * bRe;

				ExRe -= (float)(dxFi_dzSt * (fFSExRe * cRe - fFSExIm * cIm));
				ExIm -= (float)(dxFi_dzSt * (fFSExRe * cIm + fFSExIm * cRe));
				EzRe -= (float)(dxFi_dzSt * (fFSEzRe * cRe - fFSEzIm * cIm));
				EzIm -= (float)(dxFi_dzSt * (fFSEzRe * cIm + fFSEzIm * cRe));
			}
			if (DataPtrs.dzFi != 0.)
			{
				bRe = DataPtrs.ExpArrZFi[Two_iz], bIm = DataPtrs.ExpArrZFi[Two_iz_p_1];
				cRe = ExpXFiRe * bRe - ExpXFiIm * bIm; cIm = ExpXFiRe * bIm + ExpXFiIm * bRe;

				ExRe += (float)(dxFi_dzFi * (fFFExRe * cRe - fFFExIm * cIm));
				ExIm += (float)(dxFi_dzFi * (fFFExRe * cIm + fFFExIm * cRe));
				EzRe += (float)(dxFi_dzFi * (fFFEzRe * cRe - fFFEzIm * cIm));
				EzIm += (float)(dxFi_dzFi * (fFFEzRe * cIm + fFFEzIm * cRe));
			}
		}
		if (DataPtrs.dzSt != 0.)
		{
			float ExpZStRe = DataPtrs.ExpArrZSt[Two_iz], ExpZStIm = DataPtrs.ExpArrZSt[Two_iz_p_1];

			bRe = DataPtrs.FFTArrZStEx[Two_ix + TwoNz * iwfr]; bIm = DataPtrs.FFTArrZStEx[Two_ix_p_1 + TwoNz * iwfr];
			ExRe += (float)(DataPtrs.dzSt * (ExpZStRe * bRe - ExpZStIm * bIm));
			ExIm += (float)(DataPtrs.dzSt * (ExpZStRe * bIm + ExpZStIm * bRe));

			bRe = DataPtrs.FFTArrZStEz[Two_ix + TwoNz * iwfr]; bIm = DataPtrs.FFTArrZStEz[Two_ix_p_1 + TwoNz * iwfr];
			EzRe += (float)(DataPtrs.dzSt * (ExpZStRe * bRe - ExpZStIm * bIm));
			EzIm += (float)(DataPtrs.dzSt * (ExpZStRe * bIm + ExpZStIm * bRe));
		}
		if (DataPtrs.dzFi != 0.)
		{
			float ExpZFiRe = DataPtrs.ExpArrZFi[Two_iz], ExpZFiIm = DataPtrs.ExpArrZFi[Two_iz_p_1];

			bRe = DataPtrs.FFTArrZFiEx[Two_ix + TwoNz * iwfr]; bIm = DataPtrs.FFTArrZFiEx[Two_ix_p_1 + TwoNz * iwfr];
			ExRe -= (float)(DataPtrs.dzFi * (ExpZFiRe * bRe - ExpZFiIm * bIm));
			ExIm -= (float)(DataPtrs.dzFi * (ExpZFiRe * bIm + ExpZFiIm * bRe));

			bRe = DataPtrs.FFTArrZFiEz[Two_ix + TwoNz * iwfr]; bIm = DataPtrs.FFTArrZFiEz[Two_ix_p_1 + TwoNz * iwfr];
			EzRe -= (float)(DataPtrs.dzFi * (ExpZFiRe * bRe - ExpZFiIm * bIm));
			EzIm -= (float)(DataPtrs.dzFi * (ExpZFiRe * bIm + ExpZFiIm * bRe));
		}

		*tEx = ExRe; *(tEx + 1) = ExIm;
		*tEz = EzRe; *(tEz + 1) = EzIm;
    }
}

void MakeWfrEdgeCorrection_CUDA(srTSRWRadStructAccessData& RadAccessData, float* pDataEx, float* pDataEz, srTDataPtrsForWfrEdgeCorr& DataPtrs)
{
	const int bs = 256;
	dim3 blocks(RadAccessData.nz / bs + ((RadAccessData.nz & (bs - 1)) != 0), RadAccessData.nx, RadAccessData.nWfr);
	dim3 threads(bs, 1);
	MakeWfrEdgeCorrection_Kernel << <blocks, threads >> > (RadAccessData, pDataEx, pDataEz, DataPtrs);

#ifdef _DEBUG
	auto err = cudaGetLastError();
	printf("%s\r\n", cudaGetErrorString(err));
#endif
}

#endif