#pragma once

void MultiplyElFieldByPhaseLin_CUDA(double xMult, double zMult, float* pBaseRadX, float* pBaseRadZ, int nWfr, int nz, int nx, int ne, float zStart, float zStep, float xStart, float xStep);