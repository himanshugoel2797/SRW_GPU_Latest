#ifdef _OFFLOAD_GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include <stdio.h>
#include <iostream>
#include <chrono>

template <class T> __global__ void RepairSignAfter1DFFT_Kernel(T* pAfterFFT, long Nx2) {
    int ix = (blockIdx.x * blockDim.x + threadIdx.x) * 4 + 2; //Nx range
    int k = threadIdx.y;  //HowMany range

    if (ix < Nx2) {
        pAfterFFT[ix + k * Nx2] = -pAfterFFT[ix + k * Nx2];
        pAfterFFT[ix + k * Nx2 + 1] = -pAfterFFT[ix + k * Nx2 + 1];
    }
}

template <class T> __global__ void RotateDataAfter1DFFT_Kernel(T* pAfterFFT, long HowMany, long Nx2, long Nx) {
    int ix = (blockIdx.x * blockDim.x + threadIdx.x) * 2; //HalfNx range
    int k = threadIdx.y; //HowMany range


    if (ix < Nx) {
        T t1_0 = pAfterFFT[ix + Nx2 * k];
        T t1_1 = pAfterFFT[ix + Nx2 * k + 1];

        pAfterFFT[ix + Nx2 * k] = pAfterFFT[ix + Nx + Nx2 * k];
        pAfterFFT[ix + Nx2 * k + 1] = pAfterFFT[ix + Nx + Nx2 * k + 1];
        pAfterFFT[ix + Nx + Nx2 * k] = t1_0;
        pAfterFFT[ix + Nx + Nx2 * k + 1] = t1_1;
    }
}

template <class T> __global__ void RepairAndRotateAfter1DFFT_Kernel(T* pAfterFFT, long HowMany, long Nx2, long Nx) {
    int ix_cmplx_num = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = ix_cmplx_num * 2; //Nx range
    int k = threadIdx.y; //HowMany range

    T real_comp = pAfterFFT[ix + k * Nx2];
    T imag_comp = pAfterFFT[ix + k * Nx2 + 1];

    if (ix < Nx2 && ((ix_cmplx_num & 1) == 1)) {
        real_comp = -real_comp;
        imag_comp = -imag_comp;
    }

    if (ix < Nx) {
        T t1_0 = real_comp;
        T t1_1 = imag_comp;

        pAfterFFT[ix + Nx2 * k] = pAfterFFT[ix + Nx + Nx2 * k];
        pAfterFFT[ix + Nx2 * k + 1] = pAfterFFT[ix + Nx + Nx2 * k + 1];
        pAfterFFT[ix + Nx + Nx2 * k] = t1_0;
        pAfterFFT[ix + Nx + Nx2 * k + 1] = t1_1;
    }
    else {
        pAfterFFT[ix + k * Nx2] = real_comp;
        pAfterFFT[ix + k * Nx2 + 1] = imag_comp;
    }
}

template <class T> __global__ void NormalizeDataAfter1DFFT_Kernel(T* pAfterFFT, long HowMany, long Nx2, T Mult) {
    int ix = (blockIdx.x * blockDim.x + threadIdx.x) * 2; //Nx range
    int k = threadIdx.y; //HowMany range

    if (ix < Nx2) {
        pAfterFFT[ix + k * Nx2] *= Mult;
        pAfterFFT[ix + k * Nx2 + 1] *= Mult;
    }
}

template <class T> __global__ void FillArrayShift_Kernel(double t0, double tStep, long N, T* arShiftX) {
    int ix = (blockIdx.x * blockDim.x + threadIdx.x); //HalfNx range

    double t0TwoPi = t0 * 2 * CUDART_PI;
    double q = tStep * ix;

    if (ix < N) {
        if (ix == 0) {
            arShiftX[N] = 1.0;
            arShiftX[N + 1] = 0.0;
        }

        ix *= 2;
        if (ix < N - 2) {
            sincos(q * t0TwoPi, &arShiftX[N + 2 + 1 + ix], &arShiftX[N + 2 + ix]);
            arShiftX[N - 2 - ix] = arShiftX[N + 2 + ix];
            arShiftX[N - 1 - ix] = -arShiftX[N + 2 + 1 + ix];
        }

        if (ix == N - 2) {
            sincos(-q * t0TwoPi, &arShiftX[1], &arShiftX[0]);
        }
    }
}

template <class T> __global__ void TreatShift_Kernel(T* pData, long Nx2, T* tShiftX) {
    int ix = (blockIdx.x * blockDim.x + threadIdx.x) * 2; //Nx range
    int k = threadIdx.y; //HowMany range

    if (ix < Nx2) {
        T MultX_Re = tShiftX[ix];
        T MultX_Im = tShiftX[ix + 1];

        T NewRe = pData[ix + k * Nx2] * MultX_Re - pData[ix + k * Nx2 + 1] * MultX_Im;
        T NewIm = pData[ix + k * Nx2] * MultX_Im + pData[ix + k * Nx2 + 1] * MultX_Re;
        pData[ix + k * Nx2] = NewRe;
        pData[ix + k * Nx2 + 1] = NewIm;
    }
}

void RepairSignAfter1DFFT_CUDA(float* pAfterFFT, long HowMany, long Nx) {
    const int bs = 256;
    dim3 blocks(Nx / bs + ((Nx & (bs - 1)) != 0), HowMany);
    dim3 threads(bs, 1);
    RepairSignAfter1DFFT_Kernel<float> << <blocks, threads >> > (pAfterFFT, Nx * 2);

#ifdef _DEBUG
    auto err = cudaGetLastError();
    printf("%s\r\n", cudaGetErrorString(err));
#endif
}

void RotateDataAfter1DFFT_CUDA(float* pAfterFFT, long HowMany, long Nx) {
    const int bs = 256;
    dim3 blocks(Nx / (2 * bs) + ((Nx / 2 & (bs - 1)) != 0), HowMany);
    dim3 threads(bs, 1);
    RotateDataAfter1DFFT_Kernel<float> << <blocks, threads >> > (pAfterFFT, HowMany, Nx * 2, Nx);

#ifdef _DEBUG
    auto err = cudaGetLastError();
    printf("%s\r\n", cudaGetErrorString(err));
#endif
}

void RepairAndRotateDataAfter1DFFT_CUDA(float* pAfterFFT, long HowMany, long Nx) {

#ifdef _DEBUG
    {
        auto err = cudaGetLastError();
        printf("%s\r\n", cudaGetErrorString(err));
    }
#endif

    const int bs = 256;
    dim3 blocks(Nx / bs + ((Nx & (bs - 1)) != 0), HowMany);
    dim3 threads(bs, 1);
    RepairAndRotateAfter1DFFT_Kernel<float> << <blocks, threads >> > (pAfterFFT, HowMany, Nx * 2, Nx);

#ifdef _DEBUG
    auto err = cudaGetLastError();
    printf("%s\r\n", cudaGetErrorString(err));
#endif
}

void NormalizeDataAfter1DFFT_CUDA(float* pAfterFFT, long HowMany, long Nx, double Mult) {
    const int bs = 256;
    dim3 blocks(Nx / bs + ((Nx & (bs - 1)) != 0), HowMany);
    dim3 threads(bs, 1);
    NormalizeDataAfter1DFFT_Kernel<float> << <blocks, threads >> > (pAfterFFT, HowMany, Nx * 2, Mult);

#ifdef _DEBUG
    auto err = cudaGetLastError();
    printf("%s\r\n", cudaGetErrorString(err));
#endif
}

void FillArrayShift_CUDA(double t0, double tStep, long Nx, float* tShiftX) {
    const int bs = 256;
    dim3 blocks(Nx / (2 * bs) + ((Nx / 2 & (bs - 1)) != 0), 1);
    dim3 threads(bs, 1);
    FillArrayShift_Kernel<float> << <blocks, threads >> > (t0, tStep, Nx, tShiftX);

#ifdef _DEBUG
    auto err = cudaGetLastError();
    printf("%s\r\n", cudaGetErrorString(err));
#endif
}

void TreatShift_CUDA(float* pData, long HowMany, long Nx, float* tShiftX) {
    const int bs = 256;
    dim3 blocks(Nx / bs + ((Nx & (bs - 1)) != 0), HowMany);
    dim3 threads(bs, 1);
    TreatShift_Kernel<float> << <blocks, threads >> > (pData, Nx * 2, tShiftX);

#ifdef _DEBUG
    auto err = cudaGetLastError();
    printf("%s\r\n", cudaGetErrorString(err));
#endif
}

void RepairSignAfter1DFFT_CUDA(double* pAfterFFT, long HowMany, long Nx) {
    const int bs = 256;
    dim3 blocks(Nx / bs + ((Nx & (bs - 1)) != 0), HowMany);
    dim3 threads(bs, 1);
    RepairSignAfter1DFFT_Kernel<double> << <blocks, threads >> > (pAfterFFT, Nx * 2);

#ifdef _DEBUG
    auto err = cudaGetLastError();
    printf("%s\r\n", cudaGetErrorString(err));
#endif
}

void RotateDataAfter1DFFT_CUDA(double* pAfterFFT, long HowMany, long Nx) {
    const int bs = 256;
    dim3 blocks(Nx / (2 * bs) + ((Nx & (2 * bs - 1)) != 0), HowMany);
    dim3 threads(bs, 1);
    RotateDataAfter1DFFT_Kernel<double> << <blocks, threads >> > (pAfterFFT, HowMany, Nx * 2, Nx);

#ifdef _DEBUG
    auto err = cudaGetLastError();
    printf("%s\r\n", cudaGetErrorString(err));
#endif
}

void RepairAndRotateDataAfter1DFFT_CUDA(double* pAfterFFT, long HowMany, long Nx) {
    const int bs = 256;
    dim3 blocks(Nx / bs + ((Nx & (bs - 1)) != 0), HowMany);
    dim3 threads(bs, 1);
    RepairAndRotateAfter1DFFT_Kernel<double> << <blocks, threads >> > (pAfterFFT, HowMany, Nx * 2, Nx);

#ifdef _DEBUG
    auto err = cudaGetLastError();
    printf("%s\r\n", cudaGetErrorString(err));
#endif
}

void NormalizeDataAfter1DFFT_CUDA(double* pAfterFFT, long HowMany, long Nx, double Mult) {
    const int bs = 256;
    dim3 blocks(Nx / bs + ((Nx & (bs - 1)) != 0), HowMany);
    dim3 threads(bs, 1);
    NormalizeDataAfter1DFFT_Kernel<double> << <blocks, threads >> > (pAfterFFT, HowMany, Nx * 2, Mult);

#ifdef _DEBUG
    auto err = cudaGetLastError();
    printf("%s\r\n", cudaGetErrorString(err));
#endif
}

void FillArrayShift_CUDA(double t0, double tStep, long Nx, double* tShiftX) {
    const int bs = 256;
    dim3 blocks(Nx / (2 * bs) + ((Nx & (2 * bs - 1)) != 0), 1);
    dim3 threads(bs, 1);
    FillArrayShift_Kernel<double> << <blocks, threads >> > (t0, tStep, Nx, tShiftX);

#ifdef _DEBUG
    auto err = cudaGetLastError();
    printf("%s\r\n", cudaGetErrorString(err));
#endif
}

void TreatShift_CUDA(double* pData, long HowMany, long Nx, double* tShiftX) {
    const int bs = 256;
    dim3 blocks(Nx / bs + ((Nx & (bs - 1)) != 0), HowMany);
    dim3 threads(bs, 1);
    TreatShift_Kernel<double> << <blocks, threads >> > (pData, Nx * 2, tShiftX);

#ifdef _DEBUG
    auto err = cudaGetLastError();
    printf("%s\r\n", cudaGetErrorString(err));
#endif
}


template <class T> __global__ void RepairSignAfter2DFFT_Kernel(T* pAfterFFT, long Nx, long Ny) {
    int ix = (blockIdx.x * blockDim.x + threadIdx.x); //Nx range
    int iy = (blockIdx.y * blockDim.y + threadIdx.y); //Ny range

    float sx0 = 1;// - 2 * (ix % 2);
    float sy0 = 1 - 2 * (iy % 2);
    float s = sx0 * sy0;

    if (ix < Nx && iy < Ny) {
        pAfterFFT[(ix + iy * Nx) * 2] *= s;
        pAfterFFT[(ix + iy * Nx) * 2 + 1] *= s;
    }
}

template <class T> __global__ void RotateDataAfter2DFFT_Kernel(T* pAfterFFT, long HalfNx, long Nx, long HalfNy, long Ny) {
    int ix = (blockIdx.x * blockDim.x + threadIdx.x); //HalfNx range
    int iy = (blockIdx.y * blockDim.y + threadIdx.y); //HalfNy range

    if (ix < HalfNx && iy < HalfNy) {

        int idx = (ix + iy * Nx) * 2;

        long long HalfNyNx = ((long long)HalfNy) * ((long long)Nx);
        T* t1 = pAfterFFT, * t2 = pAfterFFT + (HalfNyNx + HalfNx) * 2;
        T* t3 = pAfterFFT + HalfNx * 2, * t4 = pAfterFFT + HalfNyNx * 2;

        T buf_r = t1[idx];
        T buf_im = t1[idx + 1];
        t1[idx] = t2[idx];
        t1[idx + 1] = t2[idx + 1];

        t2[idx] = buf_r;
        t2[idx + 1] = buf_im;

        buf_r = t3[idx];
        buf_im = t3[idx + 1];
        t3[idx] = t4[idx];
        t3[idx + 1] = t4[idx + 1];

        t4[idx] = buf_r;
        t4[idx + 1] = buf_im;
    }
}

template <class T> __global__ void NormalizeDataAfter2DFFT_Kernel(T* pAfterFFT, long Nx2Ny2, T Mult) {
    int ix = (blockIdx.x * blockDim.x + threadIdx.x) * 2; //Nx range

    if (ix < Nx2Ny2) {
        pAfterFFT[ix] *= Mult;
        pAfterFFT[ix + 1] *= Mult;
    }
}


void RepairSignAfter2DFFT_CUDA(float* pAfterFFT, long Nx, long Ny) {
    const int bs = 256;
    dim3 blocks(Nx / bs + ((Nx & (bs - 1)) != 0), Ny);
    dim3 threads(bs, 1);
    RepairSignAfter2DFFT_Kernel<float> << <blocks, threads >> > (pAfterFFT, Nx, Ny);
}

void RotateDataAfter2DFFT_CUDA(float* pAfterFFT, long Nx, long Ny) {
    const int bs = 256;
    dim3 blocks(Nx / (2 * bs) + ((Nx / 2 & (bs - 1)) != 0), Ny);
    dim3 threads(bs, 1);
    RotateDataAfter2DFFT_Kernel<float> << <blocks, threads >> > (pAfterFFT, Nx / 2, Nx, Ny / 2, Ny);
}

void NormalizeDataAfter2DFFT_CUDA(float* pAfterFFT, long Nx, long Ny, double Mult) {
    const int bs = 256;
    dim3 blocks((Nx * Ny) / bs + (((Nx * Ny) & (bs - 1)) != 0), 1);
    dim3 threads(bs, 1);
    NormalizeDataAfter2DFFT_Kernel<float> << <blocks, threads >> > (pAfterFFT, Nx * Ny * 2, Mult);
}

void RepairSignAfter2DFFT_CUDA(double* pAfterFFT, long Nx, long Ny) {
    const int bs = 256;
    dim3 blocks(Nx / bs + ((Nx & (bs - 1)) != 0), Ny);
    dim3 threads(bs, 1);
    RepairSignAfter2DFFT_Kernel<double> << <blocks, threads >> > (pAfterFFT, Nx, Ny);
}

void RotateDataAfter2DFFT_CUDA(double* pAfterFFT, long Nx, long Ny) {
    const int bs = 256;
    dim3 blocks(Nx / (2 * bs) + ((Nx / 2 & (bs - 1)) != 0), Ny);
    dim3 threads(bs, 1);
    RotateDataAfter2DFFT_Kernel<double> << <blocks, threads >> > (pAfterFFT, Nx / 2, Nx, Ny / 2, Ny);
}

void NormalizeDataAfter2DFFT_CUDA(double* pAfterFFT, long Nx, long Ny, double Mult) {
    const int bs = 256;
    dim3 blocks((Nx * Ny) / bs + (((Nx * Ny) & (bs - 1)) != 0), 1);
    dim3 threads(bs, 1);
    NormalizeDataAfter2DFFT_Kernel<double> << <blocks, threads >> > (pAfterFFT, Nx * Ny * 2, Mult);
}


__global__ void StokesAvgUpdateInterp_Kernel(float* pStokesArS, float* pMoreStokesArS, int nIters, int nOrder, int nStokesComp, double mult, int iSt, long xNpMeshRes, long yNpMeshRes, long eNpMeshRes, double yStartMeshRes, double yStepMeshRes, double yStartWfr, double yStepWfr, double xStartMeshRes, double xStepMeshRes, double xStartWfr, double xStepWfr, int iOfstSt, long xNpWfr, long yNpWfr, long eNpWfr)
{
    int ix = (blockIdx.x * blockDim.x + threadIdx.x); //xNpMeshRes range
    int iy = (blockIdx.y * blockDim.y + threadIdx.y); //yNpMeshRes range
    int ie = (blockIdx.z * blockDim.z + threadIdx.z); //eNpMeshRes range

    if (ix >= xNpMeshRes)
        return;
    if (iy >= yNpMeshRes)
        return;
    if (ie >= eNpMeshRes)
        return;

    long ir = iSt * yNpMeshRes * xNpMeshRes * eNpMeshRes + iy * xNpMeshRes * eNpMeshRes + ix * eNpMeshRes + ie;

    auto yMeshRes = yStartMeshRes + iy * yStepMeshRes;
    auto xMeshRes = xStartMeshRes + ix * xStepMeshRes;
    double fInterp = 0;
    int loc_ix_ofst = iOfstSt + ie;
    auto nx_ix_per = xNpWfr * eNpWfr;

    int ix0 = (int) trunc((xMeshRes - xStartWfr) / xStepWfr + 1e-09);
    if ((ix0 < 0) | (ix0 >= xNpWfr - 1)) {
        pStokesArS[ir] = pStokesArS[ir] * nIters / (nIters + 1);
        return;
    }
    int ix1 = ix0 + 1;
    auto tx = (xMeshRes - (xStartWfr + xStepWfr * ix0)) / xStepWfr;
    int iy0 = (int)trunc((yMeshRes - yStartWfr) / yStepWfr + 1e-09);
    if ((iy0 < 0) | (iy0 >= yNpWfr - 1)) {
        pStokesArS[ir] = pStokesArS[ir] * nIters / (nIters + 1);
        return;
    }


    int iy1 = iy0 + 1;
    auto ty = (yMeshRes - (yStartWfr + yStepWfr * iy0)) / yStepWfr;
    auto iy0_nx_ix_per = iy0 * nx_ix_per;
    auto iy1_nx_ix_per = iy1 * nx_ix_per;
    auto ix0_ix_per_p_ix_ofst = ix0 * eNpWfr + loc_ix_ofst;
    auto ix1_ix_per_p_ix_ofst = ix1 * eNpWfr + loc_ix_ofst;
    auto a00 = pMoreStokesArS[iy0_nx_ix_per + ix0_ix_per_p_ix_ofst];
    auto f10 = pMoreStokesArS[iy0_nx_ix_per + ix1_ix_per_p_ix_ofst];
    auto f01 = pMoreStokesArS[iy1_nx_ix_per + ix0_ix_per_p_ix_ofst];
    auto f11 = pMoreStokesArS[iy1_nx_ix_per + ix1_ix_per_p_ix_ofst];
    auto a10 = f10 - a00;
    auto a01 = f01 - a00;
    auto a11 = a00 - f01 - f10 + f11;
    fInterp = a00 + tx * (a10 + ty * a11) + ty * a01;


    pStokesArS[ir] = (pStokesArS[ir] * nIters + mult * fInterp) / (nIters + 1);
    return;
}

void StokesAvgUpdateInterp(float* pStokesArS, float* pMoreStokesArS, int nIters, int nOrder, int nStokesComp, double mult, int iSt, long xNpMeshRes, long yNpMeshRes, long eNpMeshRes, double yStartMeshRes, double yStepMeshRes, double yStartWfr, double yStepWfr, double xStartMeshRes, double xStepMeshRes, double xStartWfr, double xStepWfr, int iOfstSt, long xNpWfr, long yNpWfr, long eNpWfr)
{
    const int bs = 8;
    dim3 threads(xNpMeshRes / bs + ((xNpMeshRes & (bs - 1)) != 0), yNpMeshRes / bs + ((yNpMeshRes & (bs - 1)) != 0), eNpMeshRes);
    dim3 blocks(bs, bs, 1);
    StokesAvgUpdateInterp_Kernel << <threads, blocks >> > (pStokesArS, pMoreStokesArS, nIters, nOrder, nStokesComp, mult, iSt, xNpMeshRes, yNpMeshRes, eNpMeshRes, yStartMeshRes, yStepMeshRes, yStartWfr, yStepWfr, xStartMeshRes, xStepMeshRes, xStartWfr, xStepWfr, iOfstSt, xNpWfr, yNpWfr, eNpWfr);
}
#endif