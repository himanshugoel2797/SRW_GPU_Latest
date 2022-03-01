#ifdef _OFFLOAD_GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include <stdio.h>
#include <iostream>
#include <chrono>
#include "sroptelmgpu.h"
#include <srstraux.h>


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
        sincosf(Phase, SinPh, CosPh);

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

__device__ void GetCellDataForInterpol(float* pSt, long long PerX_Old, long long PerZ_Old, srTInterpolAuxF* tF)
{// Fills Re and Im parts of Ex or Ez
	float* pf00 = pSt; tF->f00 = *pf00;
	float* pf10 = pf00 + PerX_Old; tF->f10 = *pf10;
	float* pf20 = pf10 + PerX_Old; tF->f20 = *pf20;
	float* pf30 = pf20 + PerX_Old; tF->f30 = *pf30;

	float* pf01 = pf00 + PerZ_Old; tF->f01 = *pf01;
	float* pf11 = pf01 + PerX_Old; tF->f11 = *pf11;
	float* pf21 = pf11 + PerX_Old; tF->f21 = *pf21;
	float* pf31 = pf21 + PerX_Old; tF->f31 = *pf31;

	float* pf02 = pf01 + PerZ_Old; tF->f02 = *pf02;
	float* pf12 = pf02 + PerX_Old; tF->f12 = *pf12;
	float* pf22 = pf12 + PerX_Old; tF->f22 = *pf22;
	float* pf32 = pf22 + PerX_Old; tF->f32 = *pf32;

	float* pf03 = pf02 + PerZ_Old; tF->f03 = *pf03;
	float* pf13 = pf03 + PerX_Old; tF->f13 = *pf13;
	float* pf23 = pf13 + PerX_Old; tF->f23 = *pf23;
	float* pf33 = pf23 + PerX_Old; tF->f33 = *pf33;
	tF++;

	tF->f00 = *(++pf00); tF->f10 = *(++pf10); tF->f20 = *(++pf20); tF->f30 = *(++pf30);
	tF->f01 = *(++pf01); tF->f11 = *(++pf11); tF->f21 = *(++pf21); tF->f31 = *(++pf31);
	tF->f02 = *(++pf02); tF->f12 = *(++pf12); tF->f22 = *(++pf22); tF->f32 = *(++pf32);
	tF->f03 = *(++pf03); tF->f13 = *(++pf13); tF->f23 = *(++pf23); tF->f33 = *(++pf33);
}

__device__ void SetupCellDataI(srTInterpolAuxF* tF, srTInterpolAuxF* tI)
{
	srTInterpolAuxF* tF1 = tF + 1;

	tI->f00 = (tF->f00) * (tF->f00) + (tF1->f00) * (tF1->f00);
	tI->f10 = (tF->f10) * (tF->f10) + (tF1->f10) * (tF1->f10);
	tI->f20 = (tF->f20) * (tF->f20) + (tF1->f20) * (tF1->f20);
	tI->f30 = (tF->f30) * (tF->f30) + (tF1->f30) * (tF1->f30);

	tI->f01 = (tF->f01) * (tF->f01) + (tF1->f01) * (tF1->f01);
	tI->f11 = (tF->f11) * (tF->f11) + (tF1->f11) * (tF1->f11);
	tI->f21 = (tF->f21) * (tF->f21) + (tF1->f21) * (tF1->f21);
	tI->f31 = (tF->f31) * (tF->f31) + (tF1->f31) * (tF1->f31);

	tI->f02 = (tF->f02) * (tF->f02) + (tF1->f02) * (tF1->f02);
	tI->f12 = (tF->f12) * (tF->f12) + (tF1->f12) * (tF1->f12);
	tI->f22 = (tF->f22) * (tF->f22) + (tF1->f22) * (tF1->f22);
	tI->f32 = (tF->f32) * (tF->f32) + (tF1->f32) * (tF1->f32);

	tI->f03 = (tF->f03) * (tF->f03) + (tF1->f03) * (tF1->f03);
	tI->f13 = (tF->f13) * (tF->f13) + (tF1->f13) * (tF1->f13);
	tI->f23 = (tF->f23) * (tF->f23) + (tF1->f23) * (tF1->f23);
	tI->f33 = (tF->f33) * (tF->f33) + (tF1->f33) * (tF1->f33);

	//tI->SetUpAvg(); tI->NormalizeByAvg();

	tI->fAvg = (float)(0.0625 * (tI->f00 + tI->f10 + tI->f20 + tI->f30 + tI->f01 + tI->f11 + tI->f21 + tI->f31 + tI->f02 + tI->f12 + tI->f22 + tI->f32 + tI->f03 + tI->f13 + tI->f23 + tI->f33));
	const float CritNorm = 1.;
	if (fabs(tI->fAvg) > CritNorm)
	{
		float a = (float)(1. / tI->fAvg);
		tI->f00 *= a; tI->f10 *= a; tI->f20 *= a; tI->f30 *= a;
		tI->f01 *= a; tI->f11 *= a; tI->f21 *= a; tI->f31 *= a;
		tI->f02 *= a; tI->f12 *= a; tI->f22 *= a; tI->f32 *= a;
		tI->f03 *= a; tI->f13 *= a; tI->f23 *= a; tI->f33 *= a;
		tI->fNorm = tI->fAvg;
	}
	else tI->fNorm = 1.;
}

__device__ void SetupInterpolAux02_LowOrder(srTInterpolAuxF* pF, srTInterpolAux01* pC, srTInterpolAux02* pA)
{
	pA->Ax0z0 = pF->f00;
	pA->Ax1z0 = pC->cLAx1z0 * (pF->f10 - pF->f00);
	pA->Ax0z1 = pC->cLAx0z1 * (pF->f01 - pF->f00);
	pA->Ax1z1 = pC->cLAx1z1 * (pF->f00 - pF->f01 - pF->f10 + pF->f11);
}

__device__ int CheckForLowOrderInterp(srTInterpolAuxF* CellF, srTInterpolAuxF* CellFI, int ixRel, int izRel, srTInterpolAux01* pC, srTInterpolAux02* pA, srTInterpolAux02* pAI)
{
	if (ixRel < 0) ixRel = 0;
	if (ixRel > 2) ixRel = 2;
	if (izRel < 0) izRel = 0;
	if (izRel > 2) izRel = 2;
	srTInterpolAuxF* t = CellF;
	int iLxLz, iUxLz, iLxUz, iUxUz;
	char LowOrderCaseNoticed = 0;
	for (int i = 0; i < 2; i++)
	{
		iLxLz = (izRel << 2) + ixRel; iUxLz = iLxLz + 1;
		iLxUz = iLxLz + 4; iUxUz = iLxUz + 1;
		if ((t->f00 == 0.) || (t->f10 == 0.) || (t->f20 == 0.) || (t->f30 == 0.) ||
			(t->f01 == 0.) || (t->f11 == 0.) || (t->f21 == 0.) || (t->f31 == 0.) ||
			(t->f02 == 0.) || (t->f12 == 0.) || (t->f22 == 0.) || (t->f32 == 0.) ||
			(t->f03 == 0.) || (t->f13 == 0.) || (t->f23 == 0.) || (t->f33 == 0.))
		{
			LowOrderCaseNoticed = 1; break;
		}
		t++;
	}
	if (LowOrderCaseNoticed)
	{
		t = CellF;
		srTInterpolAuxF AuxF[2];
		srTInterpolAuxF* tAuxF = AuxF;
		for (int i = 0; i < 2; i++)
		{
			float BufF[] = { t->f00, t->f10, t->f20, t->f30, t->f01, t->f11, t->f21, t->f31, t->f02, t->f12, t->f22, t->f32, t->f03, t->f13, t->f23, t->f33 };
			tAuxF->f00 = BufF[iLxLz]; tAuxF->f10 = BufF[iUxLz]; tAuxF->f01 = BufF[iLxUz]; tAuxF->f11 = BufF[iUxUz];
			SetupInterpolAux02_LowOrder(tAuxF, pC, pA + i);
			t++; tAuxF++;
		}

		t = CellFI;
		srTInterpolAuxF AuxFI;
		float BufFI[] = { t->f00, t->f10, t->f20, t->f30, t->f01, t->f11, t->f21, t->f31, t->f02, t->f12, t->f22, t->f32, t->f03, t->f13, t->f23, t->f33 };
		AuxFI.f00 = BufFI[iLxLz]; AuxFI.f10 = BufFI[iUxLz]; AuxFI.f01 = BufFI[iLxUz]; AuxFI.f11 = BufFI[iUxUz];
		SetupInterpolAux02_LowOrder(&AuxFI, pC, pAI);
	}
	return LowOrderCaseNoticed;
}

__device__ void SetupInterpolAux02(srTInterpolAuxF* pF, srTInterpolAux01* pC, srTInterpolAux02* pA)
{
	pA->Ax0z0 = pF->f11;
	pA->Ax0z1 = (-2 * pF->f10 - 3 * pF->f11 + 6 * pF->f12 - pF->f13) * pC->cAx0z1;
	pA->Ax0z2 = (pF->f10 + pF->f12 - 2 * pF->f11) * pC->cAx0z2;
	pA->Ax0z3 = (pF->f13 - pF->f10 + 3 * (pF->f11 - pF->f12)) * pC->cAx0z3;
	pA->Ax1z0 = (-2 * pF->f01 - 3 * pF->f11 + 6 * pF->f21 - pF->f31) * pC->cAx1z0;
	pA->Ax1z1 = (4 * pF->f00 + 6 * (pF->f01 + pF->f10 - pF->f23 - pF->f32) - 12 * (pF->f02 + pF->f20) + 2 * (pF->f03 + pF->f30) + 9 * pF->f11 - 18 * (pF->f12 + pF->f21) + 3 * (pF->f13 + pF->f31) + 36 * pF->f22 + pF->f33) * pC->cAx1z1;
	pA->Ax1z2 = (-2 * (pF->f00 + pF->f02 - pF->f31) + 4 * pF->f01 - 3 * (pF->f10 + pF->f12) + 6 * (pF->f11 + pF->f20 + pF->f22) - 12 * pF->f21 - pF->f30 - pF->f32) * pC->cAx1z2;
	pA->Ax1z3 = (2 * (pF->f00 - pF->f03) + 6 * (-pF->f01 + pF->f02 - pF->f20 + pF->f23) + 3 * (pF->f10 - pF->f13 - pF->f31 + pF->f32) + 9 * (pF->f12 - pF->f11) + 18 * (pF->f21 - pF->f22) + pF->f30 - pF->f33) * pC->cAx1z3;
	pA->Ax2z0 = (pF->f01 + pF->f21 - 2 * pF->f11) * pC->cAx2z0;
	pA->Ax2z1 = (2 * (-pF->f00 + pF->f13 - pF->f20) - 3 * (pF->f21 + pF->f01) + 6 * (pF->f02 + pF->f11 + pF->f22) + 4 * pF->f10 - 12 * pF->f12 - pF->f23 - pF->f03) * pC->cAx2z1;
	pA->Ax2z2 = (pF->f00 + pF->f02 + pF->f22 + pF->f20 - 2 * (pF->f01 + pF->f10 + pF->f12 + pF->f21) + 4 * pF->f11) * pC->cAx2z2;
	pA->Ax2z3 = (-pF->f00 + pF->f03 - pF->f20 + pF->f23 + 3 * (pF->f01 - pF->f02 + pF->f21 - pF->f22) + 2 * (pF->f10 - pF->f13) + 6 * (pF->f12 - pF->f11)) * pC->cAx2z3;
	pA->Ax3z0 = (pF->f31 - pF->f01 + 3 * (pF->f11 - pF->f21)) * pC->cAx3z0;
	pA->Ax3z1 = (2 * (pF->f00 - pF->f30) + 3 * (pF->f01 - pF->f13 + pF->f23 - pF->f31) + 6 * (-pF->f02 - pF->f10 + pF->f20 + pF->f32) + 9 * (pF->f21 - pF->f11) + 18 * (pF->f12 - pF->f22) + pF->f03 - pF->f33) * pC->cAx3z1;
	pA->Ax3z2 = (pF->f30 + pF->f32 - pF->f00 - pF->f02 + 2 * (pF->f01 - pF->f31) + 3 * (pF->f10 + pF->f12 - pF->f20 - pF->f22) + 6 * (pF->f21 - pF->f11)) * pC->cAx3z2;
	pA->Ax3z3 = (pF->f00 - pF->f03 - pF->f30 + pF->f33 + 3 * (-pF->f01 + pF->f02 - pF->f10 + pF->f13 + pF->f20 - pF->f23 + pF->f31 - pF->f32) + 9 * (pF->f11 - pF->f12 - pF->f21 + pF->f22)) * pC->cAx3z3;
}

__device__ void InterpolF_LowOrder(srTInterpolAux02* A, double x, double z, float* F, int Offset)
{
	double xz = x * z;
	srTInterpolAux02* tA = A + Offset;
	for (int i = 0; i < 4 - Offset; i++)
	{
		F[i + Offset] = (float)(tA->Ax1z1 * xz + tA->Ax1z0 * x + tA->Ax0z1 * z + tA->Ax0z0);
		tA++;
	}
}

__device__ void InterpolFI_LowOrder(srTInterpolAux02* A, double x, double z, float* F, int Offset)
{
	double xz = x * z;
	srTInterpolAux02* tA = A + Offset;
	double Buf = tA->Ax1z1 * xz + tA->Ax1z0 * x + tA->Ax0z1 * z + tA->Ax0z0;
	*(F + Offset) = (float)((Buf > 0.) ? Buf : 0.);
}

__device__ void InterpolF(srTInterpolAux02* A, double x, double z, float* F, int Offset)
{
	double xE2 = x * x, xz = x * z, zE2 = z * z;
	double xE3 = xE2 * x, xE2z = xE2 * z, xzE2 = x * zE2, zE3 = zE2 * z, xE2zE2 = xE2 * zE2;
	double xE3z = xE3 * z, xE3zE2 = xE3 * zE2, xE3zE3 = xE3 * zE3, xE2zE3 = xE2 * zE3, xzE3 = x * zE3;
	srTInterpolAux02* tA = A + Offset;
	for (int i = 0; i < 4 - Offset; i++)
	{
		F[i + Offset] = (float)(tA->Ax3z3 * xE3zE3 + tA->Ax3z2 * xE3zE2 + tA->Ax3z1 * xE3z + tA->Ax3z0 * xE3 + tA->Ax2z3 * xE2zE3 + tA->Ax2z2 * xE2zE2 + tA->Ax2z1 * xE2z + tA->Ax2z0 * xE2 + tA->Ax1z3 * xzE3 + tA->Ax1z2 * xzE2 + tA->Ax1z1 * xz + tA->Ax1z0 * x + tA->Ax0z3 * zE3 + tA->Ax0z2 * zE2 + tA->Ax0z1 * z + tA->Ax0z0);
		tA++;
	}
}

__device__ void InterpolFI(srTInterpolAux02* A, double x, double z, float* F, int Offset)
{
	double xE2 = x * x, xz = x * z, zE2 = z * z;
	double xE3 = xE2 * x, xE2z = xE2 * z, xzE2 = x * zE2, zE3 = zE2 * z, xE2zE2 = xE2 * zE2;
	double xE3z = xE3 * z, xE3zE2 = xE3 * zE2, xE3zE3 = xE3 * zE3, xE2zE3 = xE2 * zE3, xzE3 = x * zE3;
	srTInterpolAux02* tA = A + Offset;
	double Buf = tA->Ax3z3 * xE3zE3 + tA->Ax3z2 * xE3zE2 + tA->Ax3z1 * xE3z + tA->Ax3z0 * xE3 + tA->Ax2z3 * xE2zE3 + tA->Ax2z2 * xE2zE2 + tA->Ax2z1 * xE2z + tA->Ax2z0 * xE2 + tA->Ax1z3 * xzE3 + tA->Ax1z2 * xzE2 + tA->Ax1z1 * xz + tA->Ax1z0 * x + tA->Ax0z3 * zE3 + tA->Ax0z2 * zE2 + tA->Ax0z1 * z + tA->Ax0z0;
	*(F + Offset) = (float)((Buf > 0.) ? Buf : 0.);
}

__device__ void ImproveReAndIm(float* pFReIm, float* pFI)
{
	float& FRe = *pFReIm, & FIm = *(pFReIm + 1);
	float AppI = FRe * FRe + FIm * FIm;
	if (AppI != 0.)
	{
		float Factor = (float)sqrt(*pFI / AppI);
		FRe *= Factor; FIm *= Factor;
	}
}

__global__ void RadResizeCore_Kernel(srTSRWRadStructAccessData OldRadAccessData, srTSRWRadStructAccessData NewRadAccessData, int iwfr, bool TreatPolCompX, bool TreatPolCompZ)
{
	const double DistAbsTol = 1.E-10;

    int ixStart = int(NewRadAccessData.AuxLong1);
    int ixEnd = int(NewRadAccessData.AuxLong2);
    int izStart = int(NewRadAccessData.AuxLong3);
    int izEnd = int(NewRadAccessData.AuxLong4);

	double xStepInvOld = 1. / OldRadAccessData.xStep;
	double zStepInvOld = 1. / OldRadAccessData.zStep;
	int nx_mi_1Old = OldRadAccessData.nx - 1;
	int nz_mi_1Old = OldRadAccessData.nz - 1;
	int nx_mi_2Old = nx_mi_1Old - 1;
	int nz_mi_2Old = nz_mi_1Old - 1;

    long long PerX_New = NewRadAccessData.ne << 1;
    long long PerZ_New = PerX_New * NewRadAccessData.nx;
    long long PerWfr_New = PerZ_New * NewRadAccessData.nz;

    long long PerX_Old = PerX_New;
    long long PerZ_Old = PerX_Old * OldRadAccessData.nx;
    long long PerWfr_Old = PerZ_Old * OldRadAccessData.nz;

    int iz = (blockIdx.x * blockDim.x + threadIdx.x) + izStart; //nz range
    int ix = (blockIdx.y * blockDim.y + threadIdx.y) + ixStart; //nx range
    int ie = (blockIdx.z * blockDim.z + threadIdx.z); //ne range

    if (ix < ixEnd && iz < izEnd && ie < NewRadAccessData.ne) 
    {
        srTInterpolAux01 InterpolAux01;
        srTInterpolAux02 InterpolAux02[4], InterpolAux02I[2];
        srTInterpolAuxF AuxF[4], AuxFI[2];
        int ixStOld, izStOld, ixStOldPrev = -1000, izStOldPrev = -1000;
        float BufF[4], BufFI[2];
        char UseLowOrderInterp_PolCompX, UseLowOrderInterp_PolCompZ;

		double zAbs = NewRadAccessData.zStart + iz * NewRadAccessData.zStep;

		char FieldShouldBeZeroedDueToZ = 0;
		if (NewRadAccessData.WfrEdgeCorrShouldBeDone)
		{
			if ((zAbs < NewRadAccessData.zWfrMin - DistAbsTol) || (zAbs > NewRadAccessData.zWfrMax + DistAbsTol)) FieldShouldBeZeroedDueToZ = 1;
		}

		int izcOld = int((zAbs - OldRadAccessData.zStart) * zStepInvOld + 1.E-06);

		double zRel = zAbs - (OldRadAccessData.zStart + izcOld * OldRadAccessData.zStep);

		if (izcOld == nz_mi_1Old) { izStOld = izcOld - 3; zRel += 2. * OldRadAccessData.zStep; }
		else if (izcOld == nz_mi_2Old) { izStOld = izcOld - 2; zRel += OldRadAccessData.zStep; }
		else if (izcOld == 0) { izStOld = izcOld; zRel -= OldRadAccessData.zStep; }
		else izStOld = izcOld - 1;

		zRel *= zStepInvOld;

		int izcOld_mi_izStOld = izcOld - izStOld;

        long long Offset_New = (iwfr * PerWfr_New) + (iz * PerZ_New) + (ix * PerX_New) + (ie * 2);
		long long ixPerX_New_p_Two_ie = ix * PerX_New + ie * 2;
		float* pEX_New = NewRadAccessData.pBaseRadX + Offset_New, * pEZ_New = NewRadAccessData.pBaseRadZ + Offset_New;

		double xAbs = NewRadAccessData.xStart + ix * NewRadAccessData.xStep;

		char FieldShouldBeZeroedDueToX = 0;
		if (NewRadAccessData.WfrEdgeCorrShouldBeDone)
		{
			if ((xAbs < NewRadAccessData.xWfrMin - DistAbsTol) || (xAbs > NewRadAccessData.xWfrMax + DistAbsTol)) FieldShouldBeZeroedDueToX = 1;
		}
		char FieldShouldBeZeroed = (FieldShouldBeZeroedDueToX || FieldShouldBeZeroedDueToZ);

		int ixcOld = int((xAbs - OldRadAccessData.xStart) * xStepInvOld + 1.E-06);
		double xRel = xAbs - (OldRadAccessData.xStart + ixcOld * OldRadAccessData.xStep);

		if (ixcOld == nx_mi_1Old) { ixStOld = ixcOld - 3; xRel += 2. * OldRadAccessData.xStep; }
		else if (ixcOld == nx_mi_2Old) { ixStOld = ixcOld - 2; xRel += OldRadAccessData.xStep; }
		else if (ixcOld == 0) { ixStOld = ixcOld; xRel -= OldRadAccessData.xStep; }
		else ixStOld = ixcOld - 1;

		xRel *= xStepInvOld;

		int ixcOld_mi_ixStOld = ixcOld - ixStOld;

		if ((izStOld != izStOldPrev) || (ixStOld != ixStOldPrev))
		{
			UseLowOrderInterp_PolCompX = 0; UseLowOrderInterp_PolCompZ = 0;

			//long TotOffsetOld = izStOld*PerZ_Old + ixStOld*PerX_Old + Two_ie;
			long long TotOffsetOld = iwfr * PerWfr_Old + izStOld * PerZ_Old + ixStOld * PerX_Old + ie * 2;

			if (TreatPolCompX)
			{
				float* pExSt_Old = OldRadAccessData.pBaseRadX + TotOffsetOld;
				GetCellDataForInterpol(pExSt_Old, PerX_Old, PerZ_Old, AuxF);

				SetupCellDataI(AuxF, AuxFI);
				UseLowOrderInterp_PolCompX = CheckForLowOrderInterp(AuxF, AuxFI, ixcOld_mi_ixStOld, izcOld_mi_izStOld, &InterpolAux01, InterpolAux02, InterpolAux02I);

				if (!UseLowOrderInterp_PolCompX)
				{
					for (int i = 0; i < 2; i++)
					{
						SetupInterpolAux02(AuxF + i, &InterpolAux01, InterpolAux02 + i);
					}
					SetupInterpolAux02(AuxFI, &InterpolAux01, InterpolAux02I);
				}
			}
			if (TreatPolCompZ)
			{
				float* pEzSt_Old = OldRadAccessData.pBaseRadZ + TotOffsetOld;
				GetCellDataForInterpol(pEzSt_Old, PerX_Old, PerZ_Old, AuxF + 2);

				SetupCellDataI(AuxF + 2, AuxFI + 1);
				UseLowOrderInterp_PolCompZ = CheckForLowOrderInterp(AuxF + 2, AuxFI + 1, ixcOld_mi_ixStOld, izcOld_mi_izStOld, &InterpolAux01, InterpolAux02 + 2, InterpolAux02I + 1);

				if (!UseLowOrderInterp_PolCompZ)
				{
					for (int i = 0; i < 2; i++)
					{
						SetupInterpolAux02(AuxF + 2 + i, &InterpolAux01, InterpolAux02 + 2 + i);
					}
					SetupInterpolAux02(AuxFI + 1, &InterpolAux01, InterpolAux02I + 1);
				}
			}

			ixStOldPrev = ixStOld; izStOldPrev = izStOld;
		}

		if (TreatPolCompX)
		{
			if (UseLowOrderInterp_PolCompX)
			{
				InterpolF_LowOrder(InterpolAux02, xRel, zRel, BufF, 0);
				InterpolFI_LowOrder(InterpolAux02I, xRel, zRel, BufFI, 0);
			}
			else
			{
				InterpolF(InterpolAux02, xRel, zRel, BufF, 0);
				InterpolFI(InterpolAux02I, xRel, zRel, BufFI, 0);
			}

			(*BufFI) *= AuxFI->fNorm;
			ImproveReAndIm(BufF, BufFI);

			if (FieldShouldBeZeroed)
			{
				*BufF = 0.; *(BufF + 1) = 0.;
			}

			*pEX_New = *BufF;
			*(pEX_New + 1) = *(BufF + 1);
		}
		if (TreatPolCompZ)
		{
			if (UseLowOrderInterp_PolCompZ)
			{
				InterpolF_LowOrder(InterpolAux02, xRel, zRel, BufF, 2);
				InterpolFI_LowOrder(InterpolAux02I, xRel, zRel, BufFI, 1);
			}
			else
			{
				InterpolF(InterpolAux02, xRel, zRel, BufF, 2);
				InterpolFI(InterpolAux02I, xRel, zRel, BufFI, 1);
			}

			(*(BufFI + 1)) *= (AuxFI + 1)->fNorm;
			ImproveReAndIm(BufF + 2, BufFI + 1);

			if (FieldShouldBeZeroed)
			{
				*(BufF + 2) = 0.; *(BufF + 3) = 0.;
			}

			*pEZ_New = *(BufF + 2);
			*(pEZ_New + 1) = *(BufF + 3);
		}
    }
}

#endif