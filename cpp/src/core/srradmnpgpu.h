/************************************************************************//**
 * File: srradmnp.h
 * Description: Various "manipulations" with Radiation data (e.g. "extraction" of Intensity from Electric Field, etc.) header
 * Project: Synchrotron Radiation Workshop
 * First release: 2000
 *
 * Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
 * All Rights Reserved
 *
 * @author O.Chubar, P.Elleaume
 * @version 1.0
 ***************************************************************************/

#ifndef __SRRADMNPGPU_H
#define __SRRADMNPGPU_H

#ifdef _OFFLOAD_GPU
#include <cuda_runtime.h>

#ifdef _USE_CUDA
#include "srradmnpgpu.cuh"
#endif

#endif

#include "omp.h"
#include "srradmnp.h"

//#include "sroptelm.h"
//#include "srstraux.h"
//#include "srerror.h"
//#include "gminterp.h"

//#include <complex>

//*************************************************************************

//void srTRadGenManip::ExtractRadiationGPU(int PolarizCompon, int Int_or_Phase, int SectID, int TransvPres, double e, double x, double z, char* pData, double* pMeth=0);

//class srTRadGenManip {
#if 0 //(defined(_WITH_OMP) and defined(_OFFLOAD_GPU))
#pragma omp declare target 
//#pragma omp device_type(any)
//#endif
	int srTRadGenManip::MutualIntensityComponentSimpleInterpolGPU(float** ExPtrs, float** ExPtrsT, float** EzPtrs, float** EzPtrsT, double InvStepRelArg, int PolCom, double iter, float* pResMI); //OC16122019
/*
	//int MutualIntensityComponentSimpleInterpol(float** ExPtrs, float** ExPtrsT, float** EzPtrs, float** EzPtrsT, double InvStepRelArg, int PolCom, int iter, float* pResMI) //OC14122019
	//int MutualIntensityComponentSimpleInterpol(float** ExPtrs, float** ExPtrsT, float** EzPtrs, float** EzPtrsT, double InvStepRelArg, int PolCom, float* pResMI)
	{//OC12092018
		float MI0[2], MI1[2];
		int res = 0;
		if(res = MutualIntensityComponent(*ExPtrs, *ExPtrsT, *EzPtrs, *EzPtrsT, PolCom, iter, MI0)) return res; //OC14122019
		if(res = MutualIntensityComponent(*(ExPtrs + 1), *(ExPtrsT + 1), *(EzPtrs + 1), *(EzPtrsT + 1), PolCom, iter, MI1)) return res;
		//if(res = MutualIntensityComponent(*ExPtrs, *ExPtrsT, *EzPtrs, *EzPtrsT, PolCom, MI0)) return res;
		//if(res = MutualIntensityComponent(*(ExPtrs + 1), *(ExPtrsT + 1), *(EzPtrs + 1), *(EzPtrsT + 1), PolCom, MI1)) return res;
		double MI0Re = *MI0, MI0Im = *(MI0 + 1);
		double MI1Re = *MI1, MI1Im = *(MI1 + 1);
		*pResMI = (float)((MI1Re - MI0Re)*InvStepRelArg + MI0Re);
		*(pResMI + 1) = (float)((MI1Im - MI0Im)*InvStepRelArg + MI0Im);
		return 0;
	}

	int srTRadGenManip::MutualIntensityComponentSimpleInterpol2DGPU(float** ExPtrs, float** ExPtrsT, float** EzPtrs, float** EzPtrsT, double Arg1, double Arg2, int PolCom, double iter, float* pResMI) //OC16122019
	//int MutualIntensityComponentSimpleInterpol2D(float** ExPtrs, float** ExPtrsT, float** EzPtrs, float** EzPtrsT, double Arg1, double Arg2, int PolCom, int iter, float* pResMI) //OC14122019
	//int MutualIntensityComponentSimpleInterpol2D(float** ExPtrs, float** ExPtrsT, float** EzPtrs, float** EzPtrsT, double Arg1, double Arg2, int PolCom, float* pResMI)
	{//OC09092018
		float MI00[2], MI10[2], MI01[2], MI11[2];
		int res = 0;
		if(res = MutualIntensityComponent(*ExPtrs, *ExPtrsT, *EzPtrs, *EzPtrsT, PolCom, iter, MI00)) return res; //OC14122019
		if(res = MutualIntensityComponent(*(ExPtrs + 1), *(ExPtrsT + 1), *(EzPtrs + 1), *(EzPtrsT + 1), PolCom, iter, MI10)) return res;
		if(res = MutualIntensityComponent(*(ExPtrs + 2), *(ExPtrsT + 2), *(EzPtrs + 2), *(EzPtrsT + 2), PolCom, iter, MI01)) return res;
		if(res = MutualIntensityComponent(*(ExPtrs + 3), *(ExPtrsT + 3), *(EzPtrs + 3), *(EzPtrsT + 3), PolCom, iter, MI11)) return res;
		//if(res = MutualIntensityComponent(*ExPtrs, *ExPtrsT, *EzPtrs, *EzPtrsT, PolCom, MI00)) return res;
		//if(res = MutualIntensityComponent(*(ExPtrs + 1), *(ExPtrsT + 1), *(EzPtrs + 1), *(EzPtrsT + 1), PolCom, MI10)) return res;
		//if(res = MutualIntensityComponent(*(ExPtrs + 2), *(ExPtrsT + 2), *(EzPtrs + 2), *(EzPtrsT + 2), PolCom, MI01)) return res;
		//if(res = MutualIntensityComponent(*(ExPtrs + 3), *(ExPtrsT + 3), *(EzPtrs + 3), *(EzPtrsT + 3), PolCom, MI11)) return res;
		double Arg1Arg2 = Arg1*Arg2;
		double MI00Re = *MI00, MI00Im = *(MI00 + 1);
		double MI10Re = *MI10, MI10Im = *(MI10 + 1);
		double MI01Re = *MI01, MI01Im = *(MI01 + 1);
		double MI11Re = *MI11, MI11Im = *(MI11 + 1);
		*pResMI = (float)((MI00Re - MI01Re - MI10Re + MI11Re)*Arg1Arg2 + (MI10Re - MI00Re)*Arg1 + (MI01Re - MI00Re)*Arg2 + MI00Re);
		*(pResMI + 1) = (float)((MI00Im - MI01Im - MI10Im + MI11Im)*Arg1Arg2 + (MI10Im - MI00Im)*Arg1 + (MI01Im - MI00Im)*Arg2 + MI00Im);
		return 0;
	}
*/

	int srTRadGenManip::MutualIntensityComponentGPU(float* pEx, float* pExT, float* pEz, float* pEzT, int PolCom, double iter, float* pResMI); //OC16122019
/*
	//int MutualIntensityComponent(float* pEx, float* pExT, float* pEz, float* pEzT, int PolCom, long long iter, float* pResMI) //OC13122019
	//int MutualIntensityComponent(float* pEx, float* pExT, float* pEz, float* pEzT, int PolCom, float* pResMI)
	{//OC09092018
	 //NOTE: This is based on M.I. definition as: E(x)E*(x'), which differs from existing definition in literature: E*(x)E(x')
	 //The two definitions are related by complex conjugation: E*(x)E(x') = (E(x)E*(x'))*
		double ExRe = 0., ExIm = 0., EzRe = 0., EzIm = 0.;
		double ExReT = 0., ExImT = 0., EzReT = 0., EzImT = 0.;
		if(EhOK) { ExRe = *pEx; ExIm = *(pEx + 1); ExReT = *pExT; ExImT = *(pExT + 1);}
		if(EvOK) { EzRe = *pEz; EzIm = *(pEz + 1); EzReT = *pEzT; EzImT = *(pEzT + 1);}
		double ReMI = 0., ImMI = 0.;

		switch(PolCom)
		{
			case 0: // Lin. Hor.
			{
				ReMI = ExRe*ExReT + ExIm*ExImT;
				ImMI = ExIm*ExReT - ExRe*ExImT;
				break;
			}
			case 1: // Lin. Vert.
			{
				ReMI = EzRe*EzReT + EzIm*EzImT;
				ImMI = EzIm*EzReT - EzRe*EzImT;
				break;
			}
			case 2: // Linear 45 deg.
			{
				double ExRe_p_EzRe = ExRe + EzRe, ExIm_p_EzIm = ExIm + EzIm;
				double ExRe_p_EzReT = ExReT + EzReT, ExIm_p_EzImT = ExImT + EzImT;
				ReMI = 0.5*(ExRe_p_EzRe*ExRe_p_EzReT + ExIm_p_EzIm*ExIm_p_EzImT);
				ImMI = 0.5*(ExIm_p_EzIm*ExRe_p_EzReT - ExRe_p_EzRe*ExIm_p_EzImT);
				break;
			}
			case 3: // Linear 135 deg.
			{
				double ExRe_mi_EzRe = ExRe - EzRe, ExIm_mi_EzIm = ExIm - EzIm;
				double ExRe_mi_EzReT = ExReT - EzReT, ExIm_mi_EzImT = ExImT - EzImT;
				ReMI = 0.5*(ExRe_mi_EzRe*ExRe_mi_EzReT + ExIm_mi_EzIm*ExIm_mi_EzImT);
				ImMI = 0.5*(ExIm_mi_EzIm*ExRe_mi_EzReT - ExRe_mi_EzRe*ExIm_mi_EzImT);
				break;
			}
			case 5: // Circ. Left //OC08092019: corrected to be in compliance with definitions for right-hand frame (x,z,s) and with corresponding definition and calculation of Stokes params
			//case 4: // Circ. Right
			{
				double ExRe_mi_EzIm = ExRe - EzIm, ExIm_p_EzRe = ExIm + EzRe;
				double ExRe_mi_EzImT = ExReT - EzImT, ExIm_p_EzReT = ExImT + EzReT;
				ReMI = 0.5*(ExRe_mi_EzIm*ExRe_mi_EzImT + ExIm_p_EzRe*ExIm_p_EzReT);
				ImMI = 0.5*(ExIm_p_EzRe*ExRe_mi_EzImT - ExRe_mi_EzIm*ExIm_p_EzReT);
				break;
			}
			case 4: // Circ. Right //OC08092019: corrected to be in compliance with definitions for right-hand frame (x,z,s) and with corresponding definition and calculation of Stokes params
			//case 5: // Circ. Left
			{
				double ExRe_p_EzIm = ExRe + EzIm, ExIm_mi_EzRe = ExIm - EzRe;
				double ExRe_p_EzImT = ExReT + EzImT, ExIm_mi_EzReT = ExImT - EzReT;
				ReMI = 0.5*(ExRe_p_EzIm*ExRe_p_EzImT + ExIm_mi_EzRe*ExIm_mi_EzReT);
				ImMI = 0.5*(ExIm_mi_EzRe*ExRe_p_EzImT - ExRe_p_EzIm*ExIm_mi_EzReT);
				break;
			}
			case -1: // s0
			{
				ReMI = ExRe*ExReT + ExIm*ExImT + EzRe*EzReT + EzIm*EzImT;
				ImMI = ExIm*ExReT - ExRe*ExImT + EzIm*EzReT - EzRe*EzImT;
				break;
			}
			case -2: // s1
			{
				ReMI = ExRe*ExReT + ExIm*ExImT - (EzRe*EzReT + EzIm*EzImT);
				ImMI = ExIm*ExReT - ExRe*ExImT - (EzIm*EzReT - EzRe*EzImT);
				break;
			}
			case -3: // s2
			{
				ReMI = ExImT*EzIm + ExIm*EzImT + ExReT*EzRe + ExRe*EzReT;
				ImMI = ExReT*EzIm - ExRe*EzImT - ExImT*EzRe + ExIm*EzReT;
				break;
			}
			case -4: // s3
			{
				ReMI = ExReT*EzIm + ExRe*EzImT - ExImT*EzRe - ExIm*EzReT;
				ImMI = ExIm*EzImT - ExImT*EzIm - ExReT*EzRe + ExRe*EzReT;
				break;
			}
			default: // total mutual intensity, same as s0
			{
				ReMI = ExRe*ExReT + ExIm*ExImT + EzRe*EzReT + EzIm*EzImT;
				ImMI = ExIm*ExReT - ExRe*ExImT + EzIm*EzReT - EzRe*EzImT;
				break;
				//return CAN_NOT_EXTRACT_MUT_INT;
			}
		}
		if(iter == 0)
		{
			*pResMI = (float)ReMI;
			*(pResMI+1) = (float)ImMI;
		}
		else if(iter > 0)
		{
			double iter_p_1 = iter + 1; //OC20012020
			//long long iter_p_1 = iter + 1;
			*pResMI = (float)(((*pResMI)*iter + ReMI)/iter_p_1);
			*(pResMI+1) = (float)(((*(pResMI+1))*iter + ImMI)/iter_p_1);
		}
		else
		{
			*pResMI += (float)ReMI;
			*(pResMI+1) += (float)ImMI;
		}
		return 0;
	}
*/
//#if (defined(_WITH_OMP) && defined(_OFFLOAD_GPU))
#pragma omp end declare target
#endif

#endif
