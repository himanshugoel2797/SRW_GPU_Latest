#ifdef _OFFLOAD_GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include "cuda_runtime_api.h"
#include <stdio.h>

#include "srtrjdat.h"
#include "srradint.h"
#include "srprgind.h"
#include "srinterf.h"
#include "srmlttsk.h"
#include "sroptelm.h"
#include "srerror.h"

#include <thrust/complex.h>

#define PI CUDART_PI

__global__ void CompTotalTrjData_FromTrj_InnerLoopCUDA(double sSt, double sEn, long long Np, double* pBtx, double* pBtz, double* pX, double* pZ, double* pIntBtxE2, double* pIntBtzE2, double* pBx, double* pBz,
	double sStp, bool VerFieldIsNotZero, bool HorFieldIsNotZero, double dxds0, double x0, double dzds0, double z0, double s0,
	double xTrjInData_Start, double xTrjInData_Step, double xTrjInData_np,
	double zTrjInData_Start, double zTrjInData_Step, double zTrjInData_np,
	double** BxPlnCf, double** BzPlnCf,
	double** BtxPlnCf, double** BtzPlnCf,
	double** xPlnCf, double** zPlnCf,
	double** IntBtx2PlnCf, double** IntBtz2PlnCf)
{
	double dxds0E2 = dxds0 * dxds0, dzds0E2 = dzds0 * dzds0;

	//int Indx;
	long long Indx;
	double sr;
	double* pB_Cf, * pBt_Cf, * pCrd_Cf, * pIntBtE2_Cf;
	int i = (blockIdx.x * blockDim.x + threadIdx.x);
	if (i >= Np) return;

	double s = sSt + i * sStp;
	if (VerFieldIsNotZero)
	{
		//Indx = int((s - xTrjInData.Start)/xTrjInData.Step);
		Indx = (long long)((s - xTrjInData_Start) / xTrjInData_Step);
		if (Indx >= xTrjInData_np - 1) Indx = xTrjInData_np - 2;
		if (Indx < 0) Indx = 0;
		sr = s - (xTrjInData_Start + xTrjInData_Step * Indx);
		if (Indx < 2) sr -= (2 - Indx) * xTrjInData_Step;
		else if (Indx < xTrjInData_np - 3);
		else if (Indx < xTrjInData_np - 2) sr += xTrjInData_Step;
		else sr += 2 * xTrjInData_Step;

		pB_Cf = *(BzPlnCf + Indx); pBt_Cf = *(BtxPlnCf + Indx); pCrd_Cf = *(xPlnCf + Indx); pIntBtE2_Cf = *(IntBtx2PlnCf + Indx);
		*(pIntBtxE2 + i) = *pIntBtE2_Cf + sr * (*(pIntBtE2_Cf + 1) + sr * (*(pIntBtE2_Cf + 2) + sr * (*(pIntBtE2_Cf + 3) + sr * (*(pIntBtE2_Cf + 4) + sr * (*(pIntBtE2_Cf + 5))))));
		*(pX + i) = *pCrd_Cf + sr * (*(pCrd_Cf + 1) + sr * (*(pCrd_Cf + 2) + sr * (*(pCrd_Cf + 3) + sr * (*(pCrd_Cf + 4) + sr * (*(pCrd_Cf + 5))))));
		*(pBtx + i) = *pBt_Cf + sr * (*(pBt_Cf + 1) + sr * (*(pBt_Cf + 2) + sr * (*(pBt_Cf + 3) + sr * (*(pBt_Cf + 4)))));
		*(pBz + i) = *pB_Cf + sr * (*(pB_Cf + 1) + sr * (*(pB_Cf + 2) + sr * (*(pB_Cf + 3))));
	}
	else
	{
		*(pBz + i) = 0.;
		double s_mi_s0 = s - s0;
		*(pBtx + i) = dxds0; *(pX + i) = x0 + dxds0 * s_mi_s0; *(pIntBtxE2 + i) = dxds0E2 * s_mi_s0;
	}
	if (HorFieldIsNotZero)
	{
		//Indx = int((s - zTrjInData.Start)/zTrjInData.Step);
		Indx = (long long)((s - zTrjInData_Start) / zTrjInData_Step);
		if (Indx >= zTrjInData_np - 1) Indx = zTrjInData_np - 2;
		if (Indx < 0) Indx = 0;
		sr = s - (zTrjInData_Start + zTrjInData_Step * Indx);
		if (Indx < 2) sr -= (2 - Indx) * zTrjInData_Step;
		else if (Indx < zTrjInData_np - 3);
		else if (Indx < zTrjInData_np - 2) sr += zTrjInData_Step;
		else sr += 2 * zTrjInData_Step;

		pB_Cf = *(BxPlnCf + Indx); pBt_Cf = *(BtzPlnCf + Indx); pCrd_Cf = *(zPlnCf + Indx); pIntBtE2_Cf = *(IntBtz2PlnCf + Indx);
		*(pIntBtzE2 + i) = *pIntBtE2_Cf + sr * (*(pIntBtE2_Cf + 1) + sr * (*(pIntBtE2_Cf + 2) + sr * (*(pIntBtE2_Cf + 3) + sr * (*(pIntBtE2_Cf + 4) + sr * (*(pIntBtE2_Cf + 5))))));
		*(pZ + i) = *pCrd_Cf + sr * (*(pCrd_Cf + 1) + sr * (*(pCrd_Cf + 2) + sr * (*(pCrd_Cf + 3) + sr * (*(pCrd_Cf + 4) + sr * (*(pCrd_Cf + 5))))));
		*(pBtz + i) = *pBt_Cf + sr * (*(pBt_Cf + 1) + sr * (*(pBt_Cf + 2) + sr * (*(pBt_Cf + 3) + sr * (*(pBt_Cf + 4)))));
		*(pBx + i) = *pB_Cf + sr * (*(pB_Cf + 1) + sr * (*(pB_Cf + 2) + sr * (*(pB_Cf + 3))));
	}
	else
	{
		*(pBx + i) = 0.;
		double s_mi_s0 = s - s0;
		*(pBtz + i) = dzds0; *(pZ + i) = z0 + dzds0 * s_mi_s0; *(pIntBtzE2 + i) = dzds0E2 * s_mi_s0;
	}
}

void srTTrjDat::CompTotalTrjData_FromTrjCUDA(double sSt, double sEn, long long Np, double* pBtx, double* pBtz, double* pX, double* pZ, double* pIntBtxE2, double* pIntBtzE2, double* pBx, double* pBz)
{
	double& dxds0 = EbmDat.dxds0, & x0 = EbmDat.x0, & dzds0 = EbmDat.dzds0, & z0 = EbmDat.z0, & s0 = EbmDat.s0;
	double dxds0E2 = dxds0 * dxds0, dzds0E2 = dzds0 * dzds0;

	double s = sSt, sStp = (Np > 1) ? (sEn - sSt) / (Np - 1) : 0.;
	//int Indx;
	long long Indx;
	double sr;
	double* pB_Cf, * pBt_Cf, * pCrd_Cf, * pIntBtE2_Cf;
	//for(long i=0; i<Np; i++)

	int bs = 256;
	if (Np < bs)
		bs = Np;
	dim3 blocks((Np) / bs + (((Np) & (bs - 1)) != 0), 1);
	dim3 threads(bs, 1);
	CompTotalTrjData_FromTrj_InnerLoopCUDA << <blocks, threads >> > (sSt, sEn, Np, pBtx, pBtz, pX, pZ, pIntBtxE2, pIntBtzE2, pBx, pBz, sStp, VerFieldIsNotZero, HorFieldIsNotZero, dxds0, x0, dzds0, z0, s0,
		xTrjInData.Start, xTrjInData.Step, xTrjInData.np,
		zTrjInData.Start, zTrjInData.Step, zTrjInData.np,
		BxPlnCf, BzPlnCf,
		BtxPlnCf, BtzPlnCf,
		xPlnCf, zPlnCf,
		IntBtx2PlnCf, IntBtz2PlnCf);
	cudaDeviceSynchronize();
}

int srTRadInt::FillNextLevelCUDA(int LevelNo, double sStart, double sEnd, long long Np)
{
	//double* BasePtr = new double[Np * 8];
	double* BasePtr = nullptr;
	cudaMallocManaged(&BasePtr, Np * 80 * sizeof(double));
	//if(BasePtr == 0) { pSend->ErrorMessage("SR::Error900"); return MEMORY_ALLOCATION_FAILURE;}
	if (BasePtr == 0) { return MEMORY_ALLOCATION_FAILURE; }

	BtxArrP[LevelNo] = BasePtr;
	BasePtr += Np; XArrP[LevelNo] = BasePtr;
	BasePtr += Np; IntBtxE2ArrP[LevelNo] = BasePtr;
	BasePtr += Np; BxArrP[LevelNo] = BasePtr;

	BasePtr += Np; BtzArrP[LevelNo] = BasePtr;
	BasePtr += Np; ZArrP[LevelNo] = BasePtr;
	BasePtr += Np; IntBtzE2ArrP[LevelNo] = BasePtr;
	BasePtr += Np; BzArrP[LevelNo] = BasePtr;

	if (Np < 512)
		TrjDatPtr->CompTotalTrjData_FromTrj(sStart, sEnd, Np, BtxArrP[LevelNo], BtzArrP[LevelNo], XArrP[LevelNo], ZArrP[LevelNo], IntBtxE2ArrP[LevelNo], IntBtzE2ArrP[LevelNo], BxArrP[LevelNo], BzArrP[LevelNo]);
	else
		TrjDatPtr->CompTotalTrjData_FromTrjCUDA(sStart, sEnd, Np, BtxArrP[LevelNo], BtzArrP[LevelNo], XArrP[LevelNo], ZArrP[LevelNo], IntBtxE2ArrP[LevelNo], IntBtzE2ArrP[LevelNo], BxArrP[LevelNo], BzArrP[LevelNo]);

	AmOfPointsOnLevel[LevelNo] = Np;
	NumberOfLevelsFilled++;
	return 0;
}

__device__ double srTTrjDat::Pol04CUDA(double s, double* c)
{
	return s * (*c + s * (*(c + 1) + s * (*(c + 2) + s * (*(c + 3)))));
}

//*************************************************************************

__device__ double srTTrjDat::Pol05CUDA(double s, double* c)
{
	return s * (*c + s * (*(c + 1) + s * (*(c + 2) + s * (*(c + 3) + s * (*(c + 4))))));
}

__device__ double srTTrjDat::Pol09CUDA(double s, double* c)
{
	return s * (*c + s * (*(c + 1) + s * (*(c + 2) + s * (*(c + 3) + s * (*(c + 4) + s * (*(c + 5) + s * (*(c + 6) + s * (*(c + 7) + s * c[8]))))))));
}

__device__ void srTTrjDat::CompTrjDataDerivedAtPoint_FromTrjCUDA(double s, double& Btx, double& Crdx, double& IntBtxE2, double& Btz, double& Crdz, double& IntBtzE2)
{
	double& dxds0 = EbmDat.dxds0, & x0 = EbmDat.x0, & dzds0 = EbmDat.dzds0, & z0 = EbmDat.z0, & s0 = EbmDat.s0;

	double sr, * pBt_Cf, * pCrd_Cf, * pIntBtE2_Cf;
	//int Indx;
	long long Indx;

	if (VerFieldIsNotZero)
	{
		//int Indx = int((s - xTrjInData.Start)*xTrjInData.InvStep); 
		Indx = (long long)((s - xTrjInData.Start) * xTrjInData.InvStep);
		if (Indx >= xTrjInData.np - 1) Indx = xTrjInData.np - 2;
		if (Indx < 0) Indx = 0;
		sr = s - (xTrjInData.Start + xTrjInData.Step * Indx);
		if (Indx < 2) sr += (Indx - 2) * xTrjInData.Step;
		else if (Indx < xTrjInData.np - 3);
		else if (Indx < xTrjInData.np - 2) sr += xTrjInData.Step;
		else sr += (xTrjInData.Step + xTrjInData.Step);

		pBt_Cf = *(BtxPlnCf + Indx); pCrd_Cf = *(xPlnCf + Indx); pIntBtE2_Cf = *(IntBtx2PlnCf + Indx);
		IntBtxE2 = *pIntBtE2_Cf + sr * (*(pIntBtE2_Cf + 1) + sr * (*(pIntBtE2_Cf + 2) + sr * (*(pIntBtE2_Cf + 3) + sr * (*(pIntBtE2_Cf + 4) + sr * (*(pIntBtE2_Cf + 5))))));
		Crdx = *pCrd_Cf + sr * (*(pCrd_Cf + 1) + sr * (*(pCrd_Cf + 2) + sr * (*(pCrd_Cf + 3) + sr * (*(pCrd_Cf + 4) + sr * (*(pCrd_Cf + 5))))));
		Btx = *pBt_Cf + sr * (*(pBt_Cf + 1) + sr * (*(pBt_Cf + 2) + sr * (*(pBt_Cf + 3) + sr * (*(pBt_Cf + 4)))));
	}
	else { double Buf = dxds0 * (s - s0); Btx = dxds0; Crdx = x0 + Buf; IntBtxE2 = dxds0 * Buf; }
	if (HorFieldIsNotZero)
	{
		//Indx = int((s - zTrjInData.Start)*zTrjInData.InvStep); 
		Indx = (long long)((s - zTrjInData.Start) * zTrjInData.InvStep);
		if (Indx >= zTrjInData.np - 1) Indx = zTrjInData.np - 2;
		if (Indx < 0) Indx = 0;
		sr = s - (zTrjInData.Start + zTrjInData.Step * Indx);
		if (Indx < 2) sr += (Indx - 2) * zTrjInData.Step;
		else if (Indx < zTrjInData.np - 3);
		else if (Indx < zTrjInData.np - 2) sr += zTrjInData.Step;
		else sr += (zTrjInData.Step + zTrjInData.Step);

		pBt_Cf = *(BtzPlnCf + Indx); pCrd_Cf = *(zPlnCf + Indx); pIntBtE2_Cf = *(IntBtz2PlnCf + Indx);
		IntBtzE2 = *pIntBtE2_Cf + sr * (*(pIntBtE2_Cf + 1) + sr * (*(pIntBtE2_Cf + 2) + sr * (*(pIntBtE2_Cf + 3) + sr * (*(pIntBtE2_Cf + 4) + sr * (*(pIntBtE2_Cf + 5))))));
		Crdz = *pCrd_Cf + sr * (*(pCrd_Cf + 1) + sr * (*(pCrd_Cf + 2) + sr * (*(pCrd_Cf + 3) + sr * (*(pCrd_Cf + 4) + sr * (*(pCrd_Cf + 5))))));
		Btz = *pBt_Cf + sr * (*(pBt_Cf + 1) + sr * (*(pBt_Cf + 2) + sr * (*(pBt_Cf + 3) + sr * (*(pBt_Cf + 4)))));
	}
	else { double Buf = dzds0 * (s - s0); Btz = dzds0; Crdz = z0 + Buf; IntBtzE2 = dzds0 * Buf; }
}

__device__ void srTTrjDat::CompTrjDataDerivedAtPointCUDA(double s, double& Btx, double& Crdx, double& IntBtxE2, double& Btz, double& Crdz, double& IntBtzE2)
{
	if (CompFromTrj) { CompTrjDataDerivedAtPoint_FromTrjCUDA(s, Btx, Crdx, IntBtxE2, Btz, Crdz, IntBtzE2); return; }

	double& dxds0 = EbmDat.dxds0, & x0 = EbmDat.x0, & dzds0 = EbmDat.dzds0, & z0 = EbmDat.z0, & s0 = EbmDat.s0;

	//int Indx = int((s - sStart)*Inv_Step); if(Indx >= LenFieldData - 1) Indx = LenFieldData - 2;
	long long Indx = (long long)((s - sStart) * Inv_Step); if (Indx >= LenFieldData - 1) Indx = LenFieldData - 2;
	double sb = sStart + Indx * sStep;
	double smsb = s - sb;
	double* Bt_CfP, * C_CfP, * IntBt2_CfP;
	if (VerFieldIsNotZero)
	{
		Bt_CfP = BtxPlnCf[Indx]; C_CfP = xPlnCf[Indx]; IntBt2_CfP = IntBtx2PlnCf[Indx];
		Btx = BtxCorr + BetaNormConst * (Pol04CUDA(smsb, Bt_CfP + 1) + *Bt_CfP);
		double BufCrd = BetaNormConst * (Pol05CUDA(smsb, C_CfP + 1) + *C_CfP);
		Crdx = xCorr + BtxCorrForX * s + BufCrd;
		IntBtxE2 = IntBtxE2Corr + BtxCorrForXe2 * s + 2. * BtxCorrForX * BufCrd + BetaNormConstE2 * (Pol09CUDA(smsb, IntBt2_CfP + 1) + *IntBt2_CfP);
	}
	else { Btx = dxds0; Crdx = x0 + dxds0 * (s - s0); IntBtxE2 = dxds0 * dxds0 * (s - s0); }
	if (HorFieldIsNotZero)
	{
		Bt_CfP = BtzPlnCf[Indx]; C_CfP = zPlnCf[Indx]; IntBt2_CfP = IntBtz2PlnCf[Indx];
		Btz = BtzCorr - BetaNormConst * (Pol04CUDA(smsb, Bt_CfP + 1) + *Bt_CfP);
		double BufCrd = -BetaNormConst * (Pol05CUDA(smsb, C_CfP + 1) + *C_CfP);
		Crdz = zCorr + BtzCorrForZ * s + BufCrd;
		IntBtzE2 = IntBtzE2Corr + BtzCorrForZe2 * s + 2. * BtzCorrForZ * BufCrd + BetaNormConstE2 * (Pol09CUDA(smsb, IntBt2_CfP + 1) + *IntBt2_CfP);
	}
	else { Btz = dzds0; Crdz = z0 + dzds0 * (s - s0); IntBtzE2 = dzds0 * dzds0 * (s - s0); }
}


__global__ void srTRadInt::RadIntegrationAudo1CUDA_Stage1(char NearField, double xObs, double yObs, double zObs, double* pX, double* pZ, double s, double sStep, double AngPhConst, double PIm10e9_d_Lamb, double GmEm2, double* pBtx, double* pBtz, double* pIntBtxE2, double* pIntBtzE2, double& Sum1XRe, double& Sum1XIm, double& Sum1ZRe, double& Sum1ZIm, double& Sum2XRe, double& Sum2XIm, double& Sum2ZRe, double& Sum2ZIm)
{
	double One_d_ymis, xObs_mi_x, zObs_mi_z, Nx, Nz;
	double Ax, Az, Ph, CosPh, SinPh, PhPrev, PhInit;
	double LongTerm, a0;

	int i = (blockIdx.x * blockDim.x + threadIdx.x);
	if (NearField)
	{
		One_d_ymis = 1. / (yObs - s);
		xObs_mi_x = xObs - *(pX + i); zObs_mi_z = zObs - *(pZ + i);
		Nx = xObs_mi_x * One_d_ymis; Nz = zObs_mi_z * One_d_ymis;

		LongTerm = *(pIntBtxE2 + i) + *(pIntBtzE2 + i);
		//a0 = LongTerm*(1. + 0.25*LongTerm*One_d_ymis) + xObs_mi_x*Nx + zObs_mi_z*Nz;
		//a = a0*One_d_ymis;
		//Ph = PIm10e9_d_Lamb*(s*GmEm2 + a0*(1 + a*(-0.25 + a*(0.125 - 0.078125*a))));
		//OC_test
		a0 = LongTerm + xObs_mi_x * Nx + zObs_mi_z * Nz;
		Ph = PIm10e9_d_Lamb * (s * GmEm2 + a0);
		//end OC_test

		Ax = (*(pBtx + i) - Nx) * One_d_ymis; Az = (*(pBtz + i) - Nz) * One_d_ymis;
	}
	else
	{
		Ph = PIm10e9_d_Lamb * (s * AngPhConst + *(pIntBtxE2 + i) + *(pIntBtzE2 + i) - (2. * xObs * (*(pX + i)) + 2. * zObs * (*(pZ + i))));
		Ax = *(pBtx + i) - xObs; Az = *(pBtz + i) - zObs;
	}
	sincos(Ph, &SinPh, &CosPh);
	Sum1XRe += Ax * CosPh; Sum1XIm += Ax * SinPh; Sum1ZRe += Az * CosPh; Sum1ZIm += Az * SinPh; s += sStep;

	i += 1;
	if (NearField)
	{
		One_d_ymis = 1. / (yObs - s);
		xObs_mi_x = xObs - *(pX + i); zObs_mi_z = zObs - *(pZ + i);
		Nx = xObs_mi_x * One_d_ymis; Nz = zObs_mi_z * One_d_ymis;

		LongTerm = *(pIntBtxE2 + i) + *(pIntBtzE2 + i);
		//a0 = LongTerm*(1. + 0.25*LongTerm*One_d_ymis) + xObs_mi_x*Nx + zObs_mi_z*Nz;
		//a = a0*One_d_ymis;
		//Ph = PIm10e9_d_Lamb*(s*GmEm2 + a0*(1 + a*(-0.25 + a*(0.125 - 0.078125*a))));
		//OC_test
		a0 = LongTerm + xObs_mi_x * Nx + zObs_mi_z * Nz;
		Ph = PIm10e9_d_Lamb * (s * GmEm2 + a0);
		//end OC_test

		Ax = (*(pBtx + i) - Nx) * One_d_ymis; Az = (*(pBtz + i) - Nz) * One_d_ymis;
	}
	else
	{
		Ph = PIm10e9_d_Lamb * (s * AngPhConst + *(pIntBtxE2 + i) + *(pIntBtzE2 + i) - (2. * xObs * (*(pX + i)) + 2. * zObs * (*(pZ + i))));
		Ax = *(pBtx + i) - xObs; Az = *(pBtz + i) - zObs;
	}
	sincos(Ph, &SinPh, &CosPh);
	Sum2XRe += Ax * CosPh; Sum2XIm += Ax * SinPh; Sum2ZRe += Az * CosPh; Sum2ZIm += Az * SinPh; s += sStep;
}

__device__ int srTRadInt::RadIntegrationAuto1CUDA(double& OutIntXRe, double& OutIntXIm, double& OutIntZRe, double& OutIntZIm, srLambXYZ ObsCoor)
{
	//const long NpOnLevelMaxNoResult = 800000000; //5000000; //2000000; // To steer; to stop computation as unsuccessful
	const long long NpOnLevelMaxNoResult = 800000000; //5000000; //2000000; // To steer; to stop computation as unsuccessful

	double ActNormConst = (DistrInfoDat.TreatLambdaAsEnergyIn_eV) ? NormalizingConst * ObsCoor.Lamb * 0.80654658E-03 : NormalizingConst / ObsCoor.Lamb;
	double PIm10e9_d_Lamb = (DistrInfoDat.TreatLambdaAsEnergyIn_eV) ? PIm10e6dEnCon * ObsCoor.Lamb : PIm10e6 * 1000. / ObsCoor.Lamb;

	const double wfe = 7. / 15.;
	const double wf1 = 16. / 15.;
	const double wf2 = 14. / 15.;
	const double wd = 1. / 15.;

	double sStart = sIntegStart;
	double sEnd = sIntegFin;

	char NearField = (DistrInfoDat.CoordOrAngPresentation == CoordPres);

	//long NpOnLevel = 5; // Must be non-even!
	long long NpOnLevel = 5; // Must be non-even!

	int result;

	double sStep = (sEnd - sStart) / (NpOnLevel - 1);
	double Ax, Az, Ph, CosPh, SinPh, PhPrev, PhInit;
	double LongTerm, a0; //, a;

	double xObs = ObsCoor.x, yObs = ObsCoor.y, zObs = ObsCoor.z, GmEm2 = TrjDatPtr->EbmDat.GammaEm2;
	double One_d_ymis, xObs_mi_x, zObs_mi_z, Nx, Nz;
	double AngPhConst, Two_xObs, Two_zObs;

	double Sum1XRe = 0., Sum1XIm = 0., Sum1ZRe = 0., Sum1ZIm = 0., Sum2XRe = 0., Sum2XIm = 0., Sum2ZRe = 0., Sum2ZIm = 0.;
	double wFxRe, wFxIm, wFzRe, wFzIm;
	int LevelNo = 0, IndxOnLevel = 0;

	double* pBtx = *BtxArrP, * pBtz = *BtzArrP, * pX = *XArrP, * pZ = *ZArrP, * pIntBtxE2 = *IntBtxE2ArrP, * pIntBtzE2 = *IntBtzE2ArrP;
	double BtxLoc, xLoc, IntBtxE2Loc, BtzLoc, zLoc, IntBtzE2Loc;

	if (NearField)
	{
		One_d_ymis = 1. / (yObs - sStart);
		xObs_mi_x = xObs - *(pX++); zObs_mi_z = zObs - *(pZ++);
		Nx = xObs_mi_x * One_d_ymis; Nz = zObs_mi_z * One_d_ymis;

		LongTerm = *(pIntBtxE2++) + *(pIntBtzE2++);
		//a0 = LongTerm*(1. + 0.25*LongTerm*One_d_ymis) + xObs_mi_x*Nx + zObs_mi_z*Nz;
		//a = a0*One_d_ymis;
		//Ph = PIm10e9_d_Lamb*(sStart*GmEm2 + a0*(1 + a*(-0.25 + a*(0.125 - 0.078125*a))));
		//OC_test
		a0 = LongTerm + xObs_mi_x * Nx + zObs_mi_z * Nz;
		Ph = PIm10e9_d_Lamb * (sStart * GmEm2 + a0);
		//end OC_test

		Ax = (*(pBtx++) - Nx) * One_d_ymis; Az = (*(pBtz++) - Nz) * One_d_ymis;
	}
	else
	{
		AngPhConst = GmEm2 + xObs * xObs + zObs * zObs; Two_xObs = 2. * xObs; Two_zObs = 2. * zObs;
		Ph = PIm10e9_d_Lamb * (sStart * AngPhConst + *(pIntBtxE2++) + *(pIntBtzE2++) - (Two_xObs * (*(pX++)) + Two_zObs * (*(pZ++))));
		Ax = *(pBtx++) - xObs; Az = *(pBtz++) - zObs;
	}
	sincos(Ph, &SinPh, &CosPh);
	wFxRe = Ax * CosPh; wFxIm = Ax * SinPh; wFzRe = Az * CosPh; wFzIm = Az * SinPh;
	PhInit = Ph;

	double s = sStart + sStep;
	IndxOnLevel = 1;
	//int AmOfLoops = (NpOnLevel - 3) >> 1;
	long long AmOfLoops = (NpOnLevel - 3) >> 1;

	//for(int i=0; i<AmOfLoops; i++)
	for (long long i = 0; i < AmOfLoops; i++)
	{
		if (NearField)
		{
			One_d_ymis = 1. / (yObs - s);
			xObs_mi_x = xObs - *(pX++); zObs_mi_z = zObs - *(pZ++);
			Nx = xObs_mi_x * One_d_ymis; Nz = zObs_mi_z * One_d_ymis;

			LongTerm = *(pIntBtxE2++) + *(pIntBtzE2++);
			//a0 = LongTerm*(1. + 0.25*LongTerm*One_d_ymis) + xObs_mi_x*Nx + zObs_mi_z*Nz;
			//a = a0*One_d_ymis;
			//Ph = PIm10e9_d_Lamb*(s*GmEm2 + a0*(1 + a*(-0.25 + a*(0.125 - 0.078125*a))));
			//OC_test
			a0 = LongTerm + xObs_mi_x * Nx + zObs_mi_z * Nz;
			Ph = PIm10e9_d_Lamb * (s * GmEm2 + a0);
			//end OC_test

			Ax = (*(pBtx++) - Nx) * One_d_ymis; Az = (*(pBtz++) - Nz) * One_d_ymis;
		}
		else
		{
			Ph = PIm10e9_d_Lamb * (s * AngPhConst + *(pIntBtxE2++) + *(pIntBtzE2++) - (Two_xObs * (*(pX++)) + Two_zObs * (*(pZ++))));
			Ax = *(pBtx++) - xObs; Az = *(pBtz++) - zObs;
		}
		sincos(Ph, &SinPh, &CosPh);
		Sum1XRe += Ax * CosPh; Sum1XIm += Ax * SinPh; Sum1ZRe += Az * CosPh; Sum1ZIm += Az * SinPh; s += sStep;

		if (NearField)
		{
			One_d_ymis = 1. / (yObs - s);
			xObs_mi_x = xObs - *(pX++); zObs_mi_z = zObs - *(pZ++);
			Nx = xObs_mi_x * One_d_ymis; Nz = zObs_mi_z * One_d_ymis;

			LongTerm = *(pIntBtxE2++) + *(pIntBtzE2++);
			//a0 = LongTerm*(1. + 0.25*LongTerm*One_d_ymis) + xObs_mi_x*Nx + zObs_mi_z*Nz;
			//a = a0*One_d_ymis;
			//Ph = PIm10e9_d_Lamb*(s*GmEm2 + a0*(1 + a*(-0.25 + a*(0.125 - 0.078125*a))));
			//OC_test
			a0 = LongTerm + xObs_mi_x * Nx + zObs_mi_z * Nz;
			Ph = PIm10e9_d_Lamb * (s * GmEm2 + a0);
			//end OC_test

			Ax = (*(pBtx++) - Nx) * One_d_ymis; Az = (*(pBtz++) - Nz) * One_d_ymis;
		}
		else
		{
			Ph = PIm10e9_d_Lamb * (s * AngPhConst + *(pIntBtxE2++) + *(pIntBtzE2++) - (Two_xObs * (*(pX++)) + Two_zObs * (*(pZ++))));
			Ax = *(pBtx++) - xObs; Az = *(pBtz++) - zObs;
		}
		sincos(Ph, &SinPh, &CosPh);
		Sum2XRe += Ax * CosPh; Sum2XIm += Ax * SinPh; Sum2ZRe += Az * CosPh; Sum2ZIm += Az * SinPh; s += sStep;
	}
	if (NearField)
	{
		One_d_ymis = 1. / (yObs - s);
		xObs_mi_x = xObs - *(pX++); zObs_mi_z = zObs - *(pZ++);
		Nx = xObs_mi_x * One_d_ymis; Nz = zObs_mi_z * One_d_ymis;

		LongTerm = *(pIntBtxE2++) + *(pIntBtzE2++);
		//a0 = LongTerm*(1. + 0.25*LongTerm*One_d_ymis) + xObs_mi_x*Nx + zObs_mi_z*Nz;
		//a = a0*One_d_ymis;
		//Ph = PIm10e9_d_Lamb*(s*GmEm2 + a0*(1 + a*(-0.25 + a*(0.125 - 0.078125*a))));
		//OC_test
		a0 = LongTerm + xObs_mi_x * Nx + zObs_mi_z * Nz;
		Ph = PIm10e9_d_Lamb * (s * GmEm2 + a0);
		//end OC_test

		Ax = (*(pBtx++) - Nx) * One_d_ymis; Az = (*(pBtz++) - Nz) * One_d_ymis;
	}
	else
	{
		Ph = PIm10e9_d_Lamb * (s * AngPhConst + *(pIntBtxE2++) + *(pIntBtzE2++) - (Two_xObs * (*(pX++)) + Two_zObs * (*(pZ++))));
		Ax = *(pBtx++) - xObs; Az = *(pBtz++) - zObs;
	}
	sincos(Ph, &SinPh, &CosPh);
	Sum1XRe += Ax * CosPh; Sum1XIm += Ax * SinPh; Sum1ZRe += Az * CosPh; Sum1ZIm += Az * SinPh; s += sStep;

	if (NearField)
	{
		One_d_ymis = 1. / (yObs - s);
		xObs_mi_x = xObs - *pX; zObs_mi_z = zObs - *pZ;
		Nx = xObs_mi_x * One_d_ymis; Nz = zObs_mi_z * One_d_ymis;

		LongTerm = *pIntBtxE2 + *pIntBtzE2;
		//a0 = LongTerm*(1. + 0.25*LongTerm*One_d_ymis) + xObs_mi_x*Nx + zObs_mi_z*Nz;
		//a = a0*One_d_ymis;
		//Ph = PIm10e9_d_Lamb*(s*GmEm2 + a0*(1 + a*(-0.25 + a*(0.125 - 0.078125*a))));
		//OC_test
		a0 = LongTerm + xObs_mi_x * Nx + zObs_mi_z * Nz;
		Ph = PIm10e9_d_Lamb * (s * GmEm2 + a0);
		//end OC_test

		Ax = (*pBtx - Nx) * One_d_ymis; Az = (*pBtz - Nz) * One_d_ymis;
	}
	else
	{
		Ph = PIm10e9_d_Lamb * (s * AngPhConst + *pIntBtxE2 + *pIntBtzE2 - (Two_xObs * (*pX) + Two_zObs * (*pZ)));
		Ax = *pBtx - xObs; Az = *pBtz - zObs;
	}
	sincos(Ph, &SinPh, &CosPh);
	wFxRe += Ax * CosPh; wFxIm += Ax * SinPh; wFzRe += Az * CosPh; wFzIm += Az * SinPh;
	wFxRe *= wfe; wFxIm *= wfe; wFzRe *= wfe; wFzIm *= wfe;

	thrust::complex<double> DifDerX = *InitDerMan - *FinDerMan;
	double wDifDerXRe = wd * DifDerX.real(), wDifDerXIm = wd * DifDerX.imag();
	thrust::complex<double> DifDerZ = *(InitDerMan + 1) - *(FinDerMan + 1);
	double wDifDerZRe = wd * DifDerZ.real(), wDifDerZIm = wd * DifDerZ.imag();

	double ActNormConst_sStep = ActNormConst * sStep;
	double IntXRe = OutIntXRe + ActNormConst_sStep * (wFxRe + wf1 * Sum1XRe + wf2 * Sum2XRe + sStep * wDifDerXRe);
	double IntXIm = OutIntXIm + ActNormConst_sStep * (wFxIm + wf1 * Sum1XIm + wf2 * Sum2XIm + sStep * wDifDerXIm);
	double IntZRe = OutIntZRe + ActNormConst_sStep * (wFzRe + wf1 * Sum1ZRe + wf2 * Sum2ZRe + sStep * wDifDerZRe);
	double IntZIm = OutIntZIm + ActNormConst_sStep * (wFzIm + wf1 * Sum1ZIm + wf2 * Sum2ZIm + sStep * wDifDerZIm);
	double SqNorm = IntXRe * IntXRe + IntXIm * IntXIm + IntZRe * IntZRe + IntZIm * IntZIm;

	//char ExtraPassForAnyCase = 0; 

	NpOnLevel--;
	char NotFinishedYet = 1;
	while (NotFinishedYet)
	{
		Sum2XRe += Sum1XRe; Sum2XIm += Sum1XIm; Sum2ZRe += Sum1ZRe; Sum2ZIm += Sum1ZIm;
		Sum1XRe = Sum1XIm = Sum1ZRe = Sum1ZIm = 0.;
		char ThisMayBeTheLastLoop = 1;
		PhPrev = PhInit;
		LevelNo++;

		double HalfStep = 0.5 * sStep;
		s = sStart + HalfStep;

		if (LevelNo <= MaxLevelForMeth_10_11)
			pBtx = BtxArrP[LevelNo]; pBtz = BtzArrP[LevelNo]; pX = XArrP[LevelNo]; pZ = ZArrP[LevelNo]; pIntBtxE2 = IntBtxE2ArrP[LevelNo]; pIntBtzE2 = IntBtzE2ArrP[LevelNo];

		double DPhMax = 0.;

		//for(long i=0; i<NpOnLevel; i++)
		for (long long i = 0; i < NpOnLevel; i++)
		{
			if (LevelNo > MaxLevelForMeth_10_11)
			{
				pBtx = &BtxLoc; pX = &xLoc; pIntBtxE2 = &IntBtxE2Loc;
				pBtz = &BtzLoc; pZ = &zLoc; pIntBtzE2 = &IntBtzE2Loc;
				TrjDatPtr->CompTrjDataDerivedAtPointCUDA(s, *pBtx, *pX, *pIntBtxE2, *pBtz, *pZ, *pIntBtzE2);
			}

			if (NearField)
			{
				One_d_ymis = 1. / (yObs - s);
				xObs_mi_x = xObs - *pX; zObs_mi_z = zObs - *pZ;
				Nx = xObs_mi_x * One_d_ymis, Nz = zObs_mi_z * One_d_ymis;

				LongTerm = *pIntBtxE2 + *pIntBtzE2;
				//a0 = LongTerm*(1. + 0.25*LongTerm*One_d_ymis) + xObs_mi_x*Nx + zObs_mi_z*Nz;
				//a = a0*One_d_ymis;
				//Ph = PIm10e9_d_Lamb*(s*GmEm2 + a0*(1 + a*(-0.25 + a*(0.125 - 0.078125*a))));
				//OC_test
				a0 = LongTerm + xObs_mi_x * Nx + zObs_mi_z * Nz;
				Ph = PIm10e9_d_Lamb * (s * GmEm2 + a0);
				//end OC_test

				Ax = (*pBtx - Nx) * One_d_ymis; Az = (*pBtz - Nz) * One_d_ymis;
			}
			else
			{
				Ph = PIm10e9_d_Lamb * (s * AngPhConst + *pIntBtxE2 + *pIntBtzE2 - (Two_xObs * (*pX) + Two_zObs * (*pZ)));
				Ax = *pBtx - xObs; Az = *pBtz - zObs;
			}

			if (LevelNo <= MaxLevelForMeth_10_11)
			{
				pBtx++; pX++; pIntBtxE2++; pBtz++; pZ++; pIntBtzE2++;
			}

			sincos(Ph, &SinPh, &CosPh);
			Sum1XRe += Ax * CosPh; Sum1XIm += Ax * SinPh; Sum1ZRe += Az * CosPh; Sum1ZIm += Az * SinPh;

			//DEBUG
			//if(::fabs(Ax) > 0.1)
			//{
			//	int aha = 1;
			//}
			//END DEBUG

			s += sStep;

			if (Ph - PhPrev > PI) ThisMayBeTheLastLoop = 0;

			double dPh = Ph - PhPrev;
			if (dPh > DPhMax) DPhMax = dPh;

			PhPrev = Ph;

			//if(i > NpOnLevel - 30)
			//{
			//	//int aha = 1;
			//	Ph *= 1.;
			//}
		}
		double ActNormConstHalfStep = ActNormConst * HalfStep;
		double LocIntXRe = OutIntXRe + ActNormConstHalfStep * (wFxRe + wf1 * Sum1XRe + wf2 * Sum2XRe + HalfStep * wDifDerXRe);
		double LocIntXIm = OutIntXIm + ActNormConstHalfStep * (wFxIm + wf1 * Sum1XIm + wf2 * Sum2XIm + HalfStep * wDifDerXIm);
		double LocIntZRe = OutIntZRe + ActNormConstHalfStep * (wFzRe + wf1 * Sum1ZRe + wf2 * Sum2ZRe + HalfStep * wDifDerZRe);
		double LocIntZIm = OutIntZIm + ActNormConstHalfStep * (wFzIm + wf1 * Sum1ZIm + wf2 * Sum2ZIm + HalfStep * wDifDerZIm);
		double LocSqNorm = LocIntXRe * LocIntXRe + LocIntXIm * LocIntXIm + LocIntZRe * LocIntZRe + LocIntZIm * LocIntZIm;

		if (ThisMayBeTheLastLoop)
		{
			double TestVal = fabs(LocSqNorm - SqNorm);
			//char SharplyGoesDown = (LocSqNorm < 0.1*SqNorm);

			char NotFinishedYetFirstTest;
			if (ProbablyTheSameLoop && (MaxFluxDensVal > 0.)) NotFinishedYetFirstTest = (TestVal > CurrentAbsPrec);
			else NotFinishedYetFirstTest = (TestVal > sIntegRelPrec * LocSqNorm);

			if (!NotFinishedYetFirstTest)
			{
				NotFinishedYet = 0;
			}
		}

		if (NotFinishedYet)
		{
			if (NpOnLevel > NpOnLevelMaxNoResult)
			{
				return CAN_NOT_COMPUTE_RADIATION_INTEGRAL;
			}
		}

		IntXRe = LocIntXRe; IntXIm = LocIntXIm; IntZRe = LocIntZRe; IntZIm = LocIntZIm;
		SqNorm = LocSqNorm;
		sStep = HalfStep; NpOnLevel *= 2;
	}

	OutIntXRe = IntXRe; OutIntXIm = IntXIm; OutIntZRe = IntZRe; OutIntZIm = IntZIm;
	if ((ProbablyTheSameLoop && (MaxFluxDensVal < SqNorm)) || !ProbablyTheSameLoop)
	{
		MaxFluxDensVal = SqNorm; CurrentAbsPrec = sIntegRelPrec * MaxFluxDensVal;
		ProbablyTheSameLoop = 1;
	}
	return 0;
}

__global__ void srTRadInt::GenRadIntegrationCUDA(float* pEx0, float* pEz0, double StepLamb, double StepX, double StepZ, long long PerX, long long PerZ)
{// Put here more functionality (switching to different methods) later
	int result;

	long long PerX = DistrInfoDat.nLamb << 1;
	long long PerZ = DistrInfoDat.nx * PerX;
	srLambXYZ ObsCoor{ DistrInfoDat.LambStart + (blockIdx.z * blockDim.z + threadIdx.z) * StepLamb, DistrInfoDat.xStart + (blockIdx.y * blockDim.y + threadIdx.y) * StepX, DistrInfoDat.yStart, DistrInfoDat.zStart + (blockIdx.x * blockDim.x + threadIdx.x) * StepZ };

	long long izPerZ = (blockIdx.x * blockDim.x + threadIdx.x) * PerZ;
	long long ixPerX = (blockIdx.y * blockDim.y + threadIdx.y) * PerX;
	int iLamb = (blockIdx.z * blockDim.z + threadIdx.z);

	long long Offset = izPerZ + ixPerX + (iLamb << 1);
	float* pEx = pEx0 + Offset, * pEz = pEz0 + Offset;

	double IntXRe = *pEx, IntXIm = *(pEx + 1);
	double IntZRe = *pEz, IntZIm = *(pEz + 1);

	RadIntegrationAuto1CUDA(IntXRe, IntXIm, IntZRe, IntZIm, ObsCoor);

	*pEx = IntXRe;
	*(pEx + 1) = IntXIm;
	*(pEz) = IntZRe;
	*(pEz + 1) = IntZIm;
}

int srTRadInt::ComputeTotalRadDistrDirectOutCUDA(srTSRWRadStructAccessData& SRWRadStructAccessData, char showProgressInd)
{
	int result = 0;
	EstimateAbsoluteTolerance();
	ProbablyTheSameLoop = 1;

	SRWRadStructAccessData.UnderSamplingX = SRWRadStructAccessData.UnderSamplingZ = 1; //OC290805
	DeallocateMemForRadDistr(); // Do not remove this

	//if(result = pSend->InitRadDistrOutFormat3(SRWRadStructAccessData, DistrInfoDat)) return result;
	//?????

	if (result = SetupRadCompStructures()) return result;
	if (DistrInfoDat.ShowPhaseOnly) return ScanPhase();

	double StepLambda = (DistrInfoDat.nLamb > 1) ? (DistrInfoDat.LambEnd - DistrInfoDat.LambStart) / (DistrInfoDat.nLamb - 1) : 0.;
	double StepX = (DistrInfoDat.nx > 1) ? (DistrInfoDat.xEnd - DistrInfoDat.xStart) / (DistrInfoDat.nx - 1) : 0.;
	double StepZ = (DistrInfoDat.nz > 1) ? (DistrInfoDat.zEnd - DistrInfoDat.zStart) / (DistrInfoDat.nz - 1) : 0.;

	//long PerX = DistrInfoDat.nLamb << 1;
	//long PerZ = DistrInfoDat.nx*PerX;
	long long PerX = DistrInfoDat.nLamb << 1;
	long long PerZ = DistrInfoDat.nx * PerX;

	float* pEx0 = SRWRadStructAccessData.pBaseRadX;
	float* pEz0 = SRWRadStructAccessData.pBaseRadZ;

	complex<double> RadIntegValues[2];
	srTEFourier EwNormDer;

	char FinalResAreSymOverX = 0, FinalResAreSymOverZ = 0;
	AnalizeFinalResultsSymmetry(FinalResAreSymOverX, FinalResAreSymOverZ);

	double xc = TrjDatPtr->EbmDat.x0;
	double zc = TrjDatPtr->EbmDat.z0;
	double xTol = StepX * 0.001, zTol = StepZ * 0.001; // To steer

	//long TotalAmOfOutPointsForInd = DistrInfoDat.nz*DistrInfoDat.nx*DistrInfoDat.nLamb;
	long long TotalAmOfOutPointsForInd = ((long long)DistrInfoDat.nz) * ((long long)DistrInfoDat.nx) * ((long long)DistrInfoDat.nLamb);
	if (FinalResAreSymOverX) TotalAmOfOutPointsForInd >>= 1;
	if (FinalResAreSymOverZ) TotalAmOfOutPointsForInd >>= 1;
	//long PointCount = 0;
	long long PointCount = 0;
	double UpdateTimeInt_s = 0.5;
	//srTCompProgressIndicator* pCompProgressInd = 0;
	//if(showProgressInd) pCompProgressInd = new srTCompProgressIndicator(TotalAmOfOutPoints, UpdateTimeInt_s);

	if (!showProgressInd) TotalAmOfOutPointsForInd = 0;
	srTCompProgressIndicator compProgressInd(TotalAmOfOutPointsForInd, UpdateTimeInt_s);

	if (NumberOfLevelsFilled == 0)
	{
		double sStart = sIntegStart;
		double sEnd = sIntegFin;
		int NpOnLevel = 5;
		double sStep = (sEnd - sStart) / (NpOnLevel - 1);

		if (result = FillNextLevelCUDA(0, sStart, sEnd, 5)) return result;

		NpOnLevel--;
		for (int LevelNo = 1; LevelNo <= MaxLevelForMeth_10_11; LevelNo++) {
			double HalfStep = 0.5 * sStep;
			double s = sStart + HalfStep;

			if (NumberOfLevelsFilled <= LevelNo) if (result = FillNextLevelCUDA(LevelNo, s, sEnd - HalfStep, NpOnLevel)) return result;
			sStep = HalfStep; NpOnLevel *= 2;
		}
	}

	//long AbsPtCount = 0;
	//ObsCoor.y = DistrInfoDat.yStart;
	//ObsCoor.z = DistrInfoDat.zStart;
	if (m_CalcResidTerminTerms > 0)
	{
		for (int iz = 0; iz < DistrInfoDat.nz; iz++)
		{
			if (FinalResAreSymOverZ) { if ((ObsCoor.z - zc) > zTol) break; }

			//long izPerZ = iz*PerZ;
			long long izPerZ = iz * PerZ;
			ObsCoor.x = DistrInfoDat.xStart;
			for (int ix = 0; ix < DistrInfoDat.nx; ix++)
			{
				if (FinalResAreSymOverX) { if ((ObsCoor.x - xc) > xTol) break; }

				//long ixPerX = ix*PerX;
				long long ixPerX = ix * PerX;
				ObsCoor.Lamb = DistrInfoDat.LambStart;
				for (int iLamb = 0; iLamb < DistrInfoDat.nLamb; iLamb++)
				{

					complex<double> ResidVal[2];
					RadIntegrationResiduals(ResidVal, &EwNormDer);

					long long Offset = izPerZ + ixPerX + (iLamb << 1);
					float* pEx = pEx0 + Offset, * pEz = pEz0 + Offset;

					*pEx = float(ResidVal[0].real());
					*(pEx + 1) = float(ResidVal[0].imag());
					*pEz = float(ResidVal[1].real());
					*(pEz + 1) = float(ResidVal[1].imag());

					ObsCoor.Lamb += StepLambda;
				}
				ObsCoor.x += StepX;
			}
			ObsCoor.z += StepZ;
		}
	}

	int bs = 1;
	dim3 threads(DistrInfoDat.nz, DistrInfoDat.nx, DistrInfoDat.nLamb);
	dim3 blocks(bs);
	GenRadIntegrationCUDA << <blocks, threads >> > (pEx0, pEz0, StepLambda, StepX, StepZ, PerX, PerZ);

	if (FinalResAreSymOverZ || FinalResAreSymOverX)
		FillInSymPartsOfResults(FinalResAreSymOverX, FinalResAreSymOverZ, SRWRadStructAccessData);

	//if(showProgressInd && (pCompProgressInd != 0)) 
	//{
	//	delete pCompProgressInd; pCompProgressInd = 0;
	//}

	//pSend->FinishRadDistrOutFormat1();
	//????
	return result;
}
#endif