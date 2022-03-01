#pragma once
#include <srradstr.h>

void TreatStronglyOscillatingTerm_CUDA(srTSRWRadStructAccessData& RadAccessData, bool TreatPolCompX, bool TreatPolCompZ, int ieStart, int ieBefEnd, double ConstRx, double ConstRz);