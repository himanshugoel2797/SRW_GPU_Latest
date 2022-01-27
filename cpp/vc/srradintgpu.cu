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

int srTRadInt::RadIntegrationAuto1CUDA(double& OutIntXRe, double& OutIntXIm, double& OutIntZRe, double& OutIntZIm, srTEFourier* pEwNormDer)
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
	if (NumberOfLevelsFilled == 0) if (result = FillNextLevel(0, sStart, sEnd, NpOnLevel)) return result;

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
	CosAndSin(Ph, CosPh, SinPh);
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
		CosAndSin(Ph, CosPh, SinPh);
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
		CosAndSin(Ph, CosPh, SinPh);
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
	CosAndSin(Ph, CosPh, SinPh);
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
	CosAndSin(Ph, CosPh, SinPh);
	wFxRe += Ax * CosPh; wFxIm += Ax * SinPh; wFzRe += Az * CosPh; wFzIm += Az * SinPh;
	wFxRe *= wfe; wFxIm *= wfe; wFzRe *= wfe; wFzIm *= wfe;

	complex<double> DifDerX = *InitDerMan - *FinDerMan;
	double wDifDerXRe = wd * DifDerX.real(), wDifDerXIm = wd * DifDerX.imag();
	complex<double> DifDerZ = *(InitDerMan + 1) - *(FinDerMan + 1);
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
		{
			if (NumberOfLevelsFilled <= LevelNo) if (result = FillNextLevel(LevelNo, s, sEnd - HalfStep, NpOnLevel)) return result;
			pBtx = BtxArrP[LevelNo]; pBtz = BtzArrP[LevelNo]; pX = XArrP[LevelNo]; pZ = ZArrP[LevelNo]; pIntBtxE2 = IntBtxE2ArrP[LevelNo]; pIntBtzE2 = IntBtzE2ArrP[LevelNo];
		}

		double DPhMax = 0.;

		//for(long i=0; i<NpOnLevel; i++)
		for (long long i = 0; i < NpOnLevel; i++)
		{
			if (LevelNo > MaxLevelForMeth_10_11)
			{
				pBtx = &BtxLoc; pX = &xLoc; pIntBtxE2 = &IntBtxE2Loc;
				pBtz = &BtzLoc; pZ = &zLoc; pIntBtzE2 = &IntBtzE2Loc;
				TrjDatPtr->CompTrjDataDerivedAtPoint(s, *pBtx, *pX, *pIntBtxE2, *pBtz, *pZ, *pIntBtzE2);
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

			CosAndSin(Ph, CosPh, SinPh);
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
			double TestVal = ::fabs(LocSqNorm - SqNorm);
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

	//long AbsPtCount = 0;
	ObsCoor.y = DistrInfoDat.yStart;
	ObsCoor.z = DistrInfoDat.zStart;
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
				if (result = GenRadIntegrationCUDA(RadIntegValues, &EwNormDer)) return result;

				//long Offset = izPerZ + ixPerX + (iLamb << 1);
				long long Offset = izPerZ + ixPerX + (iLamb << 1);
				float* pEx = pEx0 + Offset, * pEz = pEz0 + Offset;

				*pEx = float(RadIntegValues->real());
				*(pEx + 1) = float(RadIntegValues->imag());
				*pEz = float(RadIntegValues[1].real());
				*(pEz + 1) = float(RadIntegValues[1].imag());

				if (showProgressInd)
				{
					//if(result = pCompProgressInd->UpdateIndicator(PointCount++)) return result;
					if (result = compProgressInd.UpdateIndicator(PointCount++)) return result;
				}
				if (result = srYield.Check()) return result;

				ObsCoor.Lamb += StepLambda;
			}
			ObsCoor.x += StepX;
		}
		ObsCoor.z += StepZ;
	}

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