/************************************************************************//**
 * File: utigpu.h
 * Description: GPU offloading detection and control
 * Project: Synchrotron Radiation Workshop
 * First release: 2000
 *
 * Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
 * All Rights Reserved
 *
 * @author H. Goel
 * @version 1.0
 ***************************************************************************/

#ifndef __UTIGPU_H
#define __UTIGPU_H

 //*************************************************************************

class UtiGPU
{
public:
	static bool GPUAvailable(); //CheckGPUAvailable etc
	static bool GPUEnabled();
	static void SetGPUStatus(bool enabled);
};

//*************************************************************************
#endif