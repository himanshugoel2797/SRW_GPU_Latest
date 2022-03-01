#pragma once
#include <srradstr.h>
#include <srstraux.h>

void TreatStronglyOscillatingTerm_CUDA(srTSRWRadStructAccessData& RadAccessData, bool TreatPolCompX, bool TreatPolCompZ, int ieStart, int ieBefEnd, double ConstRx, double ConstRz);
void MakeWfrEdgeCorrection_CUDA(srTSRWRadStructAccessData& RadAccessData, float* pDataEx, float* pDataEz, srTDataPtrsForWfrEdgeCorr& DataPtrs);