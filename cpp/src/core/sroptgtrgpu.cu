#ifdef _OFFLOAD_GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#endif

#include <stdio.h>
#include <iostream>
#include <chrono>
#include "sroptgtr.h"

//*************************************************************************

GPU_PORTABLE void srTGenTransmission::RadPointModifierPortable(srTEXZ& EXZ, srTEFieldPtrs& EPtrs, void* pBufVars) //OC29082019
//void srTGenTransmission::RadPointModifier(srTEXZ& EXZ, srTEFieldPtrs& EPtrs)
{// e in eV; Length in m !!!
 // Operates on Coord. side !!!
	//double xRel = EXZ.x - TransvCenPoint.x, zRel = EXZ.z - TransvCenPoint.y;
	double xRel = EXZ.x, zRel = EXZ.z; //OC080311

	long Ne = 1, Nemi2 = -1;
	long iDimX = 0, iDimZ = 1;
	if (GenTransNumData.AmOfDims == 3)
	{
		//Ne = (GenTransNumData.DimSizes)[0];
		Ne = (long)((GenTransNumData.DimSizes)[0]); //OC28042019
		Nemi2 = Ne - 2;
		iDimX = 1; iDimZ = 2;
	}

	//long Nx = (GenTransNumData.DimSizes)[0], Nz = (GenTransNumData.DimSizes)[1];
	//long Nx = (GenTransNumData.DimSizes)[iDimX], Nz = (GenTransNumData.DimSizes)[iDimZ]; //OC241112
	long Nx = (long)((GenTransNumData.DimSizes)[iDimX]), Nz = (long)((GenTransNumData.DimSizes)[iDimZ]); //OC28042019
	long Nxmi2 = Nx - 2, Nzmi2 = Nz - 2;

	//double xStart = (GenTransNumData.DimStartValues)[0], zStart = (GenTransNumData.DimStartValues)[1];
	//double xStep = (GenTransNumData.DimSteps)[0], zStep = (GenTransNumData.DimSteps)[1];
	double xStart = (GenTransNumData.DimStartValues)[iDimX], zStart = (GenTransNumData.DimStartValues)[iDimZ];
	double xStep = (GenTransNumData.DimSteps)[iDimX], zStep = (GenTransNumData.DimSteps)[iDimZ];

	double xEnd = xStart + (Nx - 1) * xStep, zEnd = zStart + (Nz - 1) * zStep;

	double AbsTolX = xStep * 0.001, AbsTolZ = zStep * 0.001; // To steer
	if (OuterTransmIs == 1)
	{
		if ((xRel < xStart - AbsTolX) || (xRel > xEnd + AbsTolX) || (zRel < zStart - AbsTolZ) || (zRel > zEnd + AbsTolZ))
		{
			if (EPtrs.pExRe != 0) { *(EPtrs.pExRe) = 0.; *(EPtrs.pExIm) = 0.; }
			if (EPtrs.pEzRe != 0) { *(EPtrs.pEzRe) = 0.; *(EPtrs.pEzIm) = 0.; }
			return;
		}
	}

	double xr = 0., zr = 0.;
	double T = 1., Ph = 0.;
	//char NotExactRightEdgeX = 1, NotExactRightEdgeZ = 1;

	long ix = long((xRel - xStart) / xStep);
	if (::fabs(xRel - ((ix + 1) * xStep + xStart)) < 1.E-05 * xStep) ix++;

	//if(ix < 0) { ix = 0; xr = 0.;}
	//else if(ix > Nxmi2) { ix = Nx - 1; xr = 0.; NotExactRightEdgeX = 0;}
	//else xr = (xRel - (ix*xStep + xStart))/xStep;

	if (ix < 0) ix = 0; //OC241112
	//else if(ix > Nxmi2) ix = Nxmi2;
	//xr = (xRel - (ix*xStep + xStart))/xStep;
	else if (ix > Nxmi2) { ix = Nxmi2; xr = 1.; }
	else xr = (xRel - (ix * xStep + xStart)) / xStep;

	long iz = long((zRel - zStart) / zStep);
	if (::fabs(zRel - ((iz + 1) * zStep + zStart)) < 1.E-05 * zStep) iz++;

	//if(iz < 0) { iz = 0; zr = 0.;}
	//else if(iz > Nzmi2) { iz = Nz - 1; zr = 0.; NotExactRightEdgeZ = 0;}
	//else zr = (zRel - (iz*zStep + zStart))/zStep;

	if (iz < 0) iz = 0;
	//else if(iz > Nzmi2) iz = Nzmi2;
	//zr = (zRel - (iz*zStep + zStart))/zStep;
	else if (iz > Nzmi2) { iz = Nzmi2; zr = 1.; }
	else zr = (zRel - (iz * zStep + zStart)) / zStep;

	double xrzr = xr * zr;
	if ((GenTransNumData.AmOfDims == 2) || ((GenTransNumData.AmOfDims == 3) && (Ne == 1)))
	{
		//long zPer = Nx << 1;
		long long zPer = Nx << 1;

		//DOUBLE *p00 = (DOUBLE*)(GenTransNumData.pData) + (iz*zPer + (ix << 1));
		//DOUBLE *p10 = p00 + 2, *p01 = p00 + zPer;
		//DOUBLE *p11 = p01 + 2;
		//DOUBLE *p00p1 = p00+1, *p10p1 = p10+1, *p01p1 = p01+1, *p11p1 = p11+1;
		double* p00 = (double*)(GenTransNumData.pData) + (iz * zPer + (ix << 1)); //OC26112019 (related to SRW port to IGOR XOP8 on Mac)
		double* p10 = p00 + 2, * p01 = p00 + zPer;
		double* p11 = p01 + 2;
		double* p00p1 = p00 + 1, * p10p1 = p10 + 1, * p01p1 = p01 + 1, * p11p1 = p11 + 1;

		//double Axz = 0., Ax = 0., Az = 0., Bxz = 0., Bx = 0., Bz = 0.;
		//if(NotExactRightEdgeX && NotExactRightEdgeZ) { Axz = *p00 - *p01 - *p10 + *p11; Bxz = *p00p1 - *p01p1 - *p10p1 + *p11p1;}
		//if(NotExactRightEdgeX) { Ax = (*p10 - *p00); Bx = (*p10p1 - *p00p1);}
		//if(NotExactRightEdgeZ) { Az = (*p01 - *p00); Bz = (*p01p1 - *p00p1);}

		double Axz = *p00 - *p01 - *p10 + *p11, Bxz = *p00p1 - *p01p1 - *p10p1 + *p11p1;
		double Ax = (*p10 - *p00), Bx = (*p10p1 - *p00p1);
		double Az = (*p01 - *p00), Bz = (*p01p1 - *p00p1);

		T = Axz * xrzr + Ax * xr + Az * zr + *p00;
		Ph = Bxz * xrzr + Bx * xr + Bz * zr + *p00p1;

		//OCTEST 04032019
		//T = *p00 + Ax*xr + Az*zr;
		//Ph = *p00p1 + Bx*xr + Bz*zr;

		//OCTEST 05032019
		//T = CGenMathInterp::InterpOnRegMesh2d(EXZ.x, EXZ.z, xStart, xStep, Nx, zStart, zStep, Nz, (double*)(GenTransNumData.pData), 3, 2);
		//Ph = CGenMathInterp::InterpOnRegMesh2d(EXZ.x, EXZ.z, xStart, xStep, Nx, zStart, zStep, Nz, (double*)(GenTransNumData.pData) + 1, 3, 2);
		//END OCTEST
	}
	else if (GenTransNumData.AmOfDims == 3)
	{//bi-linear 3D interpolation
		double eStart = (GenTransNumData.DimStartValues)[0];
		double eStep = (GenTransNumData.DimSteps)[0];

		long ie = long((EXZ.e - eStart) / eStep + 1.e-10);
		if (ie < 0) ie = 0;
		else if (ie > Nemi2) ie = Nemi2;

		double er = (EXZ.e - (ie * eStep + eStart)) / eStep;
		//double erxr = er*xr, erzr = er*zr;
		//double erxrzr = erxr*zr;

		//long xPer = Ne << 1;
		//long zPer = Nx*xPer;
		long long xPer = Ne << 1;
		long long zPer = Nx * xPer;
		//DOUBLE *p000 = (DOUBLE*)(GenTransNumData.pData) + (iz*zPer + ix*xPer + (ie << 1));
		//DOUBLE *p100 = p000 + 2, *p010 = p000 + xPer, *p001 = p000 + zPer;
		//DOUBLE *p110 = p100 + xPer, *p101 = p100 + zPer, *p011 = p010 + zPer;
		//DOUBLE *p111 = p110 + zPer;
		double* p000 = (double*)(GenTransNumData.pData) + (iz * zPer + ix * xPer + (ie << 1)); //OC26112019 (related to SRW port to IGOR XOP8 on Mac)
		double* p100 = p000 + 2, * p010 = p000 + xPer, * p001 = p000 + zPer;
		double* p110 = p100 + xPer, * p101 = p100 + zPer, * p011 = p010 + zPer;
		double* p111 = p110 + zPer;

		double one_mi_er = 1. - er, one_mi_xr = 1. - xr, one_mi_zr = 1. - zr;
		double one_mi_er_one_mi_xr = one_mi_er * one_mi_xr, er_one_mi_xr = er * one_mi_xr;
		double one_mi_er_xr = one_mi_er * xr, er_xr = er * xr;
		T = ((*p000) * one_mi_er_one_mi_xr + (*p100) * er_one_mi_xr + (*p010) * one_mi_er_xr + (*p110) * er_xr) * one_mi_zr
			+ ((*p001) * one_mi_er_one_mi_xr + (*p101) * er_one_mi_xr + (*p011) * one_mi_er_xr + (*p111) * er_xr) * zr;
		Ph = ((*(p000 + 1)) * one_mi_er_one_mi_xr + (*(p100 + 1)) * er_one_mi_xr + (*(p010 + 1)) * one_mi_er_xr + (*(p110 + 1)) * er_xr) * one_mi_zr
			+ ((*(p001 + 1)) * one_mi_er_one_mi_xr + (*(p101 + 1)) * er_one_mi_xr + (*(p011 + 1)) * one_mi_er_xr + (*(p111 + 1)) * er_xr) * zr;

		// inArFunc[] = {f(x0,y0,z0),f(x1,y0,z0),f(x0,y1,z0),f(x0,y0,z1),f(x1,y1,z0),f(x1,y0,z1),f(x0,y1,z1),f(x1,y1,z1)} //function values at the corners of the cube
		   //return inArFunc[0]*one_mi_xt*one_mi_yt*one_mi_zt
		   //	+ inArFunc[1]*xt*one_mi_yt*one_mi_zt
		   //	+ inArFunc[2]*one_mi_xt*yt*one_mi_zt
		   //	+ inArFunc[3]*one_mi_xt*one_mi_yt*zt
		   //	+ inArFunc[4]*xt*yt*one_mi_zt
		   //	+ inArFunc[5]*xt*one_mi_yt*zt
		   //	+ inArFunc[6]*one_mi_xt*yt*zt
		   //	+ inArFunc[7]*xt*yt*zt;
	}

	if (OptPathOrPhase == 1) Ph *= EXZ.e * 5.0676816042E+06; // TwoPi_d_Lambda_m
	float CosPh, SinPh; 
#ifdef _OFFLOAD_GPU
	sincosf(Ph, &SinPh, &CosPh);
#else
	CosAndSin(Ph, CosPh, SinPh);
#endif
	if (EPtrs.pExRe != 0)
	{
		float NewExRe = (float)(T * ((*(EPtrs.pExRe)) * CosPh - (*(EPtrs.pExIm)) * SinPh));
		float NewExIm = (float)(T * ((*(EPtrs.pExRe)) * SinPh + (*(EPtrs.pExIm)) * CosPh));
		*(EPtrs.pExRe) = NewExRe; *(EPtrs.pExIm) = NewExIm;
	}
	if (EPtrs.pEzRe != 0)
	{
		float NewEzRe = (float)(T * ((*(EPtrs.pEzRe)) * CosPh - (*(EPtrs.pEzIm)) * SinPh));
		float NewEzIm = (float)(T * ((*(EPtrs.pEzRe)) * SinPh + (*(EPtrs.pEzIm)) * CosPh));
		*(EPtrs.pEzRe) = NewEzRe; *(EPtrs.pEzIm) = NewEzIm;
	}
}

#ifdef _OFFLOAD_GPU
int srTGenTransmission::RadPointModifierParallel(srTSRWRadStructAccessData* pRadAccessData, void* pBufVars, long pBufVarsSz) { return RadPointModifierParallelImpl<srTGenTransmission>(pRadAccessData, pBufVars, pBufVarsSz, this); } //HG03092022
#endif