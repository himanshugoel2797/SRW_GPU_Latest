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

#define PI CUDART_PI

/*void* operator new(std::size_t n) throw(std::bad_alloc)
{
	void* tmp = nullptr;
	cudaSetDevice(0);
	cudaMallocManaged(&tmp, n);
	return tmp;
}
void operator delete(void* p) throw()
{
	//cudaFree(p);
}
void* operator new[](std::size_t n) throw(std::bad_alloc)
{
	void* tmp = nullptr;
	cudaMallocManaged(&tmp, n);
	return tmp;
}
void operator delete[](void* p) throw()
{
	//cudaFree(p);
}*/

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

	TrjDatPtr->CompTotalTrjData_FromTrjCUDA(sStart, sEnd, Np, BtxArrP[LevelNo], BtzArrP[LevelNo], XArrP[LevelNo], ZArrP[LevelNo], IntBtxE2ArrP[LevelNo], IntBtzE2ArrP[LevelNo], BxArrP[LevelNo], BzArrP[LevelNo]);

	AmOfPointsOnLevel[LevelNo] = Np;
	NumberOfLevelsFilled++;
	return 0;
}

__device__
int getGlobalIdx_3D_1D() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}


__global__ void RadIntegrationAuto1_PrecomputeCUDA(
	double xStart, double StepX, 
	double yStart, 
	double zStart, double StepZ, 
	double LambStart, double StepLambda,
	srTRadIntData_t *data,
	double GammaEm2,
	double NormalizingConst,
	double PIm10e6dEnCon,
	double PIm10e6,
	int MaxLevelForMeth_10_11,
	double** BtxArrP,
	double** BtzArrP,
	double** XArrP,
	double** ZArrP,
	double** IntBtxE2ArrP,
	double** IntBtzE2ArrP,
	long long PerZ,
	long long PerX,
	char TreatLambdaAsEnergyIn_eV,
	char NearField,
	double sIntegStart,
	double sIntegFin,
	int *curPool)
{
	int execId = curPool[getGlobalIdx_3D_1D()];
	if (execId == -1)
		return;

	long long Offset = ((blockIdx.z * blockDim.z + threadIdx.z) * PerZ) + ((blockIdx.x * blockDim.x + threadIdx.x) * PerX) + (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	double ObsCoorX = xStart + (blockIdx.x * blockDim.x + threadIdx.x) * StepX;
	double ObsCoorY = yStart;
	double ObsCoorZ = zStart + (blockIdx.z * blockDim.z + threadIdx.z) * StepZ;
	double ObsCoorLamb = LambStart + (blockIdx.y * blockDim.y + threadIdx.y) * StepLambda;

	//const long NpOnLevelMaxNoResult = 800000000; //5000000; //2000000; // To steer; to stop computation as unsuccessful
	const long long NpOnLevelMaxNoResult = 800000000; //5000000; //2000000; // To steer; to stop computation as unsuccessful

	double ActNormConst = (TreatLambdaAsEnergyIn_eV) ? NormalizingConst * ObsCoorLamb * 0.80654658E-03 : NormalizingConst / ObsCoorLamb;
	double PIm10e9_d_Lamb = (TreatLambdaAsEnergyIn_eV) ? PIm10e6dEnCon * ObsCoorLamb : PIm10e6 * 1000. / ObsCoorLamb;

	const double wfe = 7. / 15.;
	const double wf1 = 16. / 15.;
	const double wf2 = 14. / 15.;
	const double wd = 1. / 15.;

	double sStart = sIntegStart;
	double sEnd = sIntegFin;

	//long NpOnLevel = 5; // Must be non-even!
	long long NpOnLevel = 5; // Must be non-even!

	int result;
	//if (NumberOfLevelsFilled == 0) if (result = FillNextLevel(0, sStart, sEnd, NpOnLevel)) return;

	double sStep = (sEnd - sStart) / (NpOnLevel - 1);
	double Ax, Az, Ph, CosPh, SinPh, PhPrev, PhInit;
	double LongTerm, a0; //, a;


	double xObs = ObsCoorX, yObs = ObsCoorY, zObs = ObsCoorZ, GmEm2 = GammaEm2;
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
	sincos(Ph, &SinPh, &CosPh);//CosAndSin(Ph, CosPh, SinPh);
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
		sincos(Ph, &SinPh, &CosPh);//CosAndSin(Ph, CosPh, SinPh);
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
		sincos(Ph, &SinPh, &CosPh);//CosAndSin(Ph, CosPh, SinPh);
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
	sincos(Ph, &SinPh, &CosPh);//CosAndSin(Ph, CosPh, SinPh);
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
	sincos(Ph, &SinPh, &CosPh);//CosAndSin(Ph, CosPh, SinPh);
	wFxRe += Ax * CosPh; wFxIm += Ax * SinPh; wFzRe += Az * CosPh; wFzIm += Az * SinPh;
	wFxRe *= wfe; wFxIm *= wfe; wFzRe *= wfe; wFzIm *= wfe;

	double wDifDerXRe = data[execId].wDifDerXRe, wDifDerXIm = data[execId].wDifDerXIm, wDifDerZRe = data[execId].wDifDerZRe, wDifDerZIm = data[execId].wDifDerZIm;

	double ActNormConst_sStep = ActNormConst * sStep;
	double IntXRe = data[execId].IntXRe + ActNormConst_sStep * (wFxRe + wf1 * Sum1XRe + wf2 * Sum2XRe + sStep * wDifDerXRe);
	double IntXIm = data[execId].IntXIm + ActNormConst_sStep * (wFxIm + wf1 * Sum1XIm + wf2 * Sum2XIm + sStep * wDifDerXIm);
	double IntZRe = data[execId].IntZRe + ActNormConst_sStep * (wFzRe + wf1 * Sum1ZRe + wf2 * Sum2ZRe + sStep * wDifDerZRe);
	double IntZIm = data[execId].IntZIm + ActNormConst_sStep * (wFzIm + wf1 * Sum1ZIm + wf2 * Sum2ZIm + sStep * wDifDerZIm);
	double SqNorm = IntXRe * IntXRe + IntXIm * IntXIm + IntZRe * IntZRe + IntZIm * IntZIm;

	Sum2XRe += Sum1XRe; Sum2XIm += Sum1XIm; Sum2ZRe += Sum1ZRe; Sum2ZIm += Sum1ZIm;
	Sum1XRe = Sum1XIm = Sum1ZRe = Sum1ZIm = 0.;
	//char ExtraPassForAnyCase = 0; 
	//PhInit, sStep, sStart, sEnd, Sum2XRe, Sum2XIm, Sum2ZRe, Sum2ZIm, wFxRe, wFxIm, wFzRe, wFzIm, wDifDerXRe, wDifDerXIm, wDifDerZRe, wDifDerZIm
	//pBtx, pBtz, pX, pZ, pIntBtxE2, pIntBtzE2
	data[execId].PhInit = PhInit;
	data[execId].sStart = sStart;
	data[execId].sEnd = sEnd;
	data[execId].Sum2XRe = Sum2XRe;
	data[execId].Sum2XIm = Sum2XIm;
	data[execId].Sum2ZRe = Sum2ZRe;
	data[execId].Sum2ZIm = Sum2ZIm;
	data[execId].SqNorm = SqNorm;
	data[execId].wFxRe = wFxRe;
	data[execId].wFxIm = wFxIm;
	data[execId].wFzRe = wFzRe;
	data[execId].wFzIm = wFzIm;

	data[execId].pBtx = pBtx;
	data[execId].pBtz = pBtz;
	data[execId].pX = pX;
	data[execId].pZ = pZ;
	data[execId].pIntBtxE2 = pIntBtxE2;
	data[execId].pIntBtzE2 = pIntBtzE2;

	data[execId].ObsCoorX = ObsCoorX;
	data[execId].ObsCoorY = ObsCoorY;
	data[execId].ObsCoorZ = ObsCoorZ;
	data[execId].ObsCoorLamb = ObsCoorLamb;
	data[execId].Offset = Offset;
}

__global__ void RadIntegrationAuto1_ComputeMainCUDA(int LevelNo, char NearField,
	double xStart, double StepX,
	double yStart,
	double zStart, double StepZ,
	double LambStart, double StepLambda,
	srTRadIntData_t *data,
	srTWfrSmp DistrInfoDat,
	double GammaEm2,
	double NormalizingConst,
	double PIm10e6dEnCon,
	double PIm10e6,
	int MaxLevelForMeth_10_11,
	double **BtxArrP,
	double **BtzArrP,
	double **XArrP,
	double **ZArrP,
	double **IntBtxE2ArrP,
	double **IntBtzE2ArrP,
	int *curPool,
	double MaxFluxDensVal, 
	double CurrentAbsPrec, 
	double sIntegRelPrec,
	int* outPool, 
	int* activeCount, 
	float* xOut, 
	float* zOut)
{
	int execId = curPool[getGlobalIdx_3D_1D()];
	if (execId == -1)
		return;

	double ObsCoorX = data[execId].ObsCoorX;
	double ObsCoorY = data[execId].ObsCoorY;
	double ObsCoorZ = data[execId].ObsCoorZ;
	double ObsCoorLamb = data[execId].ObsCoorLamb;

	double PhInit = data[execId].PhInit;
	double sStart = data[execId].sStart;
	double sEnd = data[execId].sEnd;
	double Sum1XRe = 0, Sum1XIm = 0, Sum1ZRe = 0, Sum1ZIm = 0;
	double Sum2XRe = data[execId].Sum2XRe, Sum2XIm = data[execId].Sum2XIm, Sum2ZRe = data[execId].Sum2ZRe, Sum2ZIm = data[execId].Sum2ZIm;
	double wFxRe = data[execId].wFxRe, wFxIm = data[execId].wFxIm, wFzRe = data[execId].wFzRe, wFzIm = data[execId].wFzIm;
	double wDifDerXRe = data[execId].wDifDerXRe, wDifDerXIm = data[execId].wDifDerXIm, wDifDerZRe = data[execId].wDifDerZRe, wDifDerZIm = data[execId].wDifDerZIm;

	double *pBtx = data[execId].pBtx, *pBtz = data[execId].pBtz;
	double* pX = data[execId].pX, * pZ = data[execId].pZ;
	double* pIntBtxE2 = data[execId].pIntBtxE2, * pIntBtzE2 = data[execId].pIntBtzE2;

	double xObs = ObsCoorX, yObs = ObsCoorY, zObs = ObsCoorZ, GmEm2 = GammaEm2;
	double ActNormConst = (DistrInfoDat.TreatLambdaAsEnergyIn_eV) ? NormalizingConst * ObsCoorLamb * 0.80654658E-03 : NormalizingConst / ObsCoorLamb;
	double PIm10e9_d_Lamb = (DistrInfoDat.TreatLambdaAsEnergyIn_eV) ? PIm10e6dEnCon * ObsCoorLamb : PIm10e6 * 1000. / ObsCoorLamb;
	double AngPhConst = GmEm2 + xObs * xObs + zObs * zObs, Two_xObs = 2. * xObs, Two_zObs = 2. * zObs;

	int NpOnLevel = 4 * (1 << (LevelNo - 1));
	char ThisMayBeTheLastLoop = 1;
	double PhPrev = PhInit;

	double sStep = (sEnd - sStart) / NpOnLevel;
	double HalfStep = 0.5 * sStep;
	double s = sStart + HalfStep;

	const double wfe = 7. / 15.;
	const double wf1 = 16. / 15.;
	const double wf2 = 14. / 15.;
	const double wd = 1. / 15.;

	if (LevelNo <= MaxLevelForMeth_10_11)
	{
		pBtx = BtxArrP[LevelNo]; pBtz = BtzArrP[LevelNo]; pX = XArrP[LevelNo]; pZ = ZArrP[LevelNo]; pIntBtxE2 = IntBtxE2ArrP[LevelNo]; pIntBtzE2 = IntBtzE2ArrP[LevelNo];
	}

	double DPhMax = 0.;
	double One_d_ymis, xObs_mi_x, zObs_mi_z, Nx, Nz, Ph, Ax, Az, CosPh, SinPh;
	double LongTerm, a0; //, a;

	//for(long i=0; i<NpOnLevel; i++)
	for (long long i = 0; i < NpOnLevel; i++)
	{
		//if (LevelNo > MaxLevelForMeth_10_11)
		//{
		//	pBtx = &BtxLoc; pX = &xLoc; pIntBtxE2 = &IntBtxE2Loc;
		//	pBtz = &BtzLoc; pZ = &zLoc; pIntBtzE2 = &IntBtzE2Loc;
		//	TrjDatPtr->CompTrjDataDerivedAtPoint(s, *pBtx, *pX, *pIntBtxE2, *pBtz, *pZ, *pIntBtzE2);
		//}

		if (NearField)
		{
			One_d_ymis = 1. / (yObs - s);
			xObs_mi_x = xObs - *pX; zObs_mi_z = zObs - *pZ;
			Nx = xObs_mi_x * One_d_ymis, Nz = zObs_mi_z * One_d_ymis;

			double LongTerm = *pIntBtxE2 + *pIntBtzE2;
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

		sincos(Ph, &SinPh, &CosPh);//CosAndSin(Ph, CosPh, SinPh);
		//if(execId == 0)
		//	printf("Ph: %f CosPh: %f SinPh: %f\r\n", Ph, CosPh, SinPh);
		Sum1XRe += Ax * CosPh; Sum1XIm += Ax * SinPh; Sum1ZRe += Az * CosPh; Sum1ZIm += Az * SinPh;

		//if (execId == 0)
		//	printf("Sum1XRe: %f Sum1XIm: %f Sum1ZRe: %f Sum1ZIm: %f\r\n", Sum1XRe, Sum1XIm, Sum1ZRe, Sum1ZIm);
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
	double LocIntXRe = /*data[execId].IntXRe*/ + ActNormConstHalfStep * (wFxRe + wf1 * Sum1XRe + wf2 * Sum2XRe + HalfStep * wDifDerXRe);
	double LocIntXIm = /*data[execId].IntXIm*/ + ActNormConstHalfStep * (wFxIm + wf1 * Sum1XIm + wf2 * Sum2XIm + HalfStep * wDifDerXIm);
	double LocIntZRe = /*data[execId].IntZRe*/ + ActNormConstHalfStep * (wFzRe + wf1 * Sum1ZRe + wf2 * Sum2ZRe + HalfStep * wDifDerZRe);
	double LocIntZIm = /*data[execId].IntZIm*/ + ActNormConstHalfStep * (wFzIm + wf1 * Sum1ZIm + wf2 * Sum2ZIm + HalfStep * wDifDerZIm);
	double LocSqNorm = LocIntXRe * LocIntXRe + LocIntXIm * LocIntXIm + LocIntZRe * LocIntZRe + LocIntZIm * LocIntZIm;

		//if (data[execId].Offset == 48)
		//	printf("Sum1XRe: %f Sum2XRe: %f P3: %f\r\n", ActNormConstHalfStep * Sum1XRe * wf1, ActNormConstHalfStep * Sum2XRe * wf2, ActNormConstHalfStep * HalfStep * wDifDerXRe);
		//if (data[execId].Offset == 48)
		//	printf("LocIntXRe: %f LocIntXIm: %f LocIntZRe: %f LocIntZIm: %f ActNormConstHalfStep: %f wFxRe: %f HalfStep: %f wDifDerXRe: %f\r\n",
		//		LocIntXRe, LocIntXIm, LocIntZRe, LocIntZIm, ActNormConstHalfStep, wFxRe, HalfStep, wDifDerXRe);

	Sum2XRe += Sum1XRe; Sum2XIm += Sum1XIm; Sum2ZRe += Sum1ZRe; Sum2ZIm += Sum1ZIm;
	data[execId].Sum2XRe = Sum2XRe;
	data[execId].Sum2XIm = Sum2XIm;
	data[execId].Sum2ZRe = Sum2ZRe;
	data[execId].Sum2ZIm = Sum2ZIm;
	data[execId].ThisMayBeTheLastLoop = ThisMayBeTheLastLoop;

	//LocIntXRe, LocIntXIm, LocIntZRe, LocIntZIm, LocSqNorm
	data[execId].LocIntXRe = LocIntXRe;
	data[execId].LocIntXIm = LocIntXIm;
	data[execId].LocIntZRe = LocIntZRe;
	data[execId].LocIntZIm = LocIntZIm;
	data[execId].LocSqNorm = LocSqNorm;

	//double LocSqNorm = data[execId].LocSqNorm;
	double SqNorm = data[execId].SqNorm;

	bool NotFinishedYet = true;
	if (ThisMayBeTheLastLoop)
	{
		double TestVal = fabs(LocSqNorm - SqNorm);
		//char SharplyGoesDown = (LocSqNorm < 0.1*SqNorm);

		char NotFinishedYetFirstTest;
		if (MaxFluxDensVal > 0.) NotFinishedYetFirstTest = (TestVal > CurrentAbsPrec);
		else NotFinishedYetFirstTest = (TestVal > sIntegRelPrec * LocSqNorm);

		if (!NotFinishedYetFirstTest)
		{
			NotFinishedYet = 0;
		}
	}

	if (NotFinishedYet)
	{
		int nIndex = atomicAdd(activeCount, 1);
		outPool[nIndex] = execId;
		//	if (NpOnLevel > NpOnLevelMaxNoResult)
		//	{
				//return CAN_NOT_COMPUTE_RADIATION_INTEGRAL;
		//	}
	}
	else {

		//long long offset = data[execId].Offset;
		//xOut[offset] = (float)LocIntXRe;
		//xOut[offset + 1] = (float)LocIntXIm;
		//zOut[offset] = (float)LocIntZRe;
		//zOut[offset + 1] = (float)LocIntZIm;
		//data[execId].IntXRe = LocIntXRe; data[execId].IntXIm = LocIntXIm; data[execId].IntZRe = LocIntZRe; data[execId].IntZIm = LocIntZIm;
	}

	//data[execId].SqNorm = LocSqNorm;

}

__global__ void RadIntegrationAuto1_CheckPrecisionCUDA(double MaxFluxDensVal, double CurrentAbsPrec, double sIntegRelPrec, srTRadIntData_t *data, int *curPool, int *outPool, int *activeCount, float *xOut, float *zOut) {
	
	int execId = curPool[getGlobalIdx_3D_1D()];
	curPool[getGlobalIdx_3D_1D()] = -1;
	if (execId == -1)
		return;

	double LocSqNorm = data[execId].LocSqNorm;
	double SqNorm = data[execId].SqNorm;

	bool NotFinishedYet = true;
	if (data[execId].ThisMayBeTheLastLoop)
	{
		double TestVal = fabs(LocSqNorm - SqNorm);
		//char SharplyGoesDown = (LocSqNorm < 0.1*SqNorm);

		char NotFinishedYetFirstTest;
		if (MaxFluxDensVal > 0.) NotFinishedYetFirstTest = (TestVal > CurrentAbsPrec);
		else NotFinishedYetFirstTest = (TestVal > sIntegRelPrec * LocSqNorm);

		if (!NotFinishedYetFirstTest)
		{
			NotFinishedYet = 0;
		}
	}

	if (NotFinishedYet)
	{
		int nIndex = atomicAdd(activeCount, 1);
		outPool[nIndex] = execId;
	//	if (NpOnLevel > NpOnLevelMaxNoResult)
	//	{
			//return CAN_NOT_COMPUTE_RADIATION_INTEGRAL;
	//	}
	}
	else {

		long long offset = data[execId].Offset;
		xOut[offset] = (float)data[execId].LocIntXRe;
		xOut[offset + 1] = (float)data[execId].LocIntXIm;
		zOut[offset] = (float)data[execId].LocIntZRe;
		zOut[offset + 1] = (float)data[execId].LocIntZIm;
	}
	
	data[execId].IntXRe = data[execId].LocIntXRe; data[execId].IntXIm = data[execId].LocIntXIm; data[execId].IntZRe = data[execId].LocIntZRe; data[execId].IntZIm = data[execId].LocIntZIm;
	data[execId].SqNorm = data[execId].LocSqNorm;
}

void srTRadInt::GenRadIntegrationCUDA(srTEFourier* EwNormDer, long long nx, long long nz, long long nlamb, long long perX, long long perZ, double xStart, double yStart, double zStart, double lambStart, double xStep, double zStep, double lambStep, float* oSol0, float* oSol1)
{
	bool WorkLeft = true;
	int bs = 1;
	//1 - Perform precomputation steps in parallel
	int result;
	if (result = FillNextLevelCUDA(0, sIntegStart, sIntegFin, 5)) return;

	int len = nx;
	while (len % bs != 0)len++;
	int nx_len = len / bs;
	len *= nz * nlamb;

	//srTRadIntData_t* execData = nullptr;
	//cudaMallocManaged(&execData, sizeof(srTRadIntData_t) * len);
	int* activeIds = nullptr;
	cudaMallocManaged(&activeIds, len * sizeof(int));
	int* activeIds_2 = nullptr;
	cudaMallocManaged(&activeIds_2, len * sizeof(int));
	
	//const double wd = 1. / 15.;
	//complex<double> DifDerX = *InitDerMan - *FinDerMan;
	//double wDifDerXRe = wd * DifDerX.real(), wDifDerXIm = wd * DifDerX.imag();
	//complex<double> DifDerZ = *(InitDerMan + 1) - *(FinDerMan + 1);
	//double wDifDerZRe = wd * DifDerZ.real(), wDifDerZIm = wd * DifDerZ.imag();

	//Populate array of all executions
	for (int i = 0; i < len; i++) {
		if (i < nx * nz * nlamb)
			activeIds[i] = i;
		else
			activeIds[i] = -1;

		activeIds_2[i] = -1;
		//execData[i].wDifDerXRe = wDifDerXRe;
		//execData[i].wDifDerXIm = wDifDerXIm;
		//execData[i].wDifDerZRe = wDifDerZRe;
		//execData[i].wDifDerZIm = wDifDerZIm;
	}

	{
		dim3 blocks(nx_len, nlamb, nz);
		dim3 threads(bs, 1);
		RadIntegrationAuto1_PrecomputeCUDA << <blocks, threads >> > (xStart, xStep, yStart, zStart, zStep, lambStart, lambStep, execData,
			TrjDatPtr->EbmDat.GammaEm2,
			NormalizingConst,
			PIm10e6dEnCon,
			PIm10e6,
			MaxLevelForMeth_10_11,
			BtxArrP,
			BtzArrP,
			XArrP,
			ZArrP,
			IntBtxE2ArrP,
			IntBtzE2ArrP,
			perZ,
			perX,
			DistrInfoDat.TreatLambdaAsEnergyIn_eV,
			(DistrInfoDat.CoordOrAngPresentation == CoordPres),
			sIntegStart,
			sIntegFin,
			activeIds);
	}

	int* activeCount = nullptr;
	auto err = cudaMallocManaged(&activeCount, sizeof(int));
	cudaDeviceSynchronize();

	int LevelNo = 1;
	int NpOnLevel = 4;
	while (WorkLeft) {
		
		*activeCount = 0;
		
		//Fill current level if needed
		double sStart = sIntegStart;
		double sEnd = sIntegFin;
		double sStep = (sEnd - sStart) / (NpOnLevel);
		double HalfStep = 0.5 * sStep;
		if (LevelNo <= MaxLevelForMeth_10_11)
		{
			if (NumberOfLevelsFilled <= LevelNo) FillNextLevelCUDA(LevelNo, sStart + HalfStep, sEnd - HalfStep, NpOnLevel);
		}
		printf("%s\r\n", cudaGetErrorString(cudaGetLastError()));

		//2 - Compute current level for all active points
		{
			dim3 blocks(nx_len, nlamb, nz);
			dim3 threads(bs, 1);
			RadIntegrationAuto1_ComputeMainCUDA << <blocks, threads >> > (LevelNo, (DistrInfoDat.CoordOrAngPresentation == CoordPres),
				xStart, xStep,
				yStart,
				zStart, zStep,
				lambStart, lambStep,
				execData,
				DistrInfoDat,
				TrjDatPtr->EbmDat.GammaEm2,
				NormalizingConst,
				PIm10e6dEnCon,
				PIm10e6,
				MaxLevelForMeth_10_11,
				BtxArrP,
				BtzArrP,
				XArrP,
				ZArrP,
				IntBtxE2ArrP,
				IntBtzE2ArrP,
				activeIds,
				MaxFluxDensVal, 
				CurrentAbsPrec, 
				sIntegRelPrec, 
				activeIds_2, 
				activeCount, 
				oSol0, oSol1);
		}
		printf("%s\r\n", cudaGetErrorString(cudaGetLastError()));
		//cudaDeviceSynchronize();
		//3 - Reduce list of executions down to points which need more work
		{
			//dim3 blocks(nx_len, nlamb, nz);
			//dim3 threads(bs, 1);
			//RadIntegrationAuto1_CheckPrecisionCUDA<<<blocks, threads>>>(MaxFluxDensVal, CurrentAbsPrec, sIntegRelPrec, execData, activeIds, activeIds_2, activeCount, oSol0, oSol1);
		}

		//swap the two activity queues
		int* tmp = activeIds_2;
		activeIds_2 = activeIds;
		activeIds = tmp;

		cudaDeviceSynchronize();
		if (*activeCount > 0)
			WorkLeft = true;
		else
			WorkLeft = false;

		//CompactList
		//5 - Goto step 2 if executions are left
		LevelNo++;
		NpOnLevel *= 2;
	}
}
