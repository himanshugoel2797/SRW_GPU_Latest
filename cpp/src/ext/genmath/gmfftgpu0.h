#pragma once

void RepairSignAfter1DFFT_CUDA(float* pAfterFFT, long HowMany, long Nx);
void RotateDataAfter1DFFT_CUDA(float* pAfterFFT, long HowMany, long Nx);
void RepairAndRotateDataAfter1DFFT_CUDA(float* pAfterFFT, long HowMany, long Nx);
void NormalizeDataAfter1DFFT_CUDA(float* pAfterFFT, long HowMany, long Nx, double Mult);
void FillArrayShift_CUDA(double t0, double tStep, long Nx, float* tShiftX);
void TreatShift_CUDA(float* pData, long HowMany, long Nx, float* tShiftX);

void RepairSignAfter1DFFT_CUDA(double* pAfterFFT, long HowMany, long Nx);
void RotateDataAfter1DFFT_CUDA(double* pAfterFFT, long HowMany, long Nx);
void RepairAndRotateDataAfter1DFFT_CUDA(double* pAfterFFT, long HowMany, long Nx);
void NormalizeDataAfter1DFFT_CUDA(double* pAfterFFT, long HowMany, long Nx, double Mult);
void FillArrayShift_CUDA(double t0, double tStep, long Nx, double* tShiftX);
void TreatShift_CUDA(double* pData, long HowMany, long Nx, double* tShiftX);

void RepairSignAfter2DFFT_CUDA(float* pAfterFFT, long Nx, long Ny, long howMany);
void RotateDataAfter2DFFT_CUDA(float* pAfterFFT, long Nx, long Ny, long howMany);
void NormalizeDataAfter2DFFT_CUDA(float* pAfterFFT, long Nx, long Ny, long howMany, double Mult);

void RepairSignAfter2DFFT_CUDA(double* pAfterFFT, long Nx, long Ny, long howMany);
void RotateDataAfter2DFFT_CUDA(double* pAfterFFT, long Nx, long Ny, long howMany);
void NormalizeDataAfter2DFFT_CUDA(double* pAfterFFT, long Nx, long Ny, long howMany, double Mult);