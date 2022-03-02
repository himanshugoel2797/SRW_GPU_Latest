SOFT_DEV_DIR=	../
SRW_SRC_DIR=	$(SOFT_DEV_DIR)/src
SRW_SRC_GEN_DIR=	$(SRW_SRC_DIR)/core
SRW_SRC_LIB_DIR=	$(SRW_SRC_DIR)/lib
SRW_SRC_GENESIS_DIR=	$(SRW_SRC_DIR)/ext/genesis/genesis_july08
SH_SRC_PARSE_DIR=	$(SRW_SRC_DIR)/ext/auxparse
SH_SRC_GEN_MATH_DIR=	$(SRW_SRC_DIR)/ext/genmath
LIB_DIR=	$(SOFT_DEV_DIR)/../ext_lib

MODE ?= 0

#CC  = gcc
#CXX = g++
CC  = cc
CXX = c++

SRW_SRC_DEF=	-D_GNU_SOURCE -D__USE_XOPEN2K8 -DFFTW_ENABLE_FLOAT -D_GM_WITHOUT_BASE -DSRWLIB_STATIC -DNO_TIMER -DANSI_DECLARATORS -DTRILIBRARY -DLINUX
CFLAGS=	-O3 -fPIC -g -I$(SRW_SRC_GEN_DIR) -I$(SRW_SRC_LIB_DIR) -I$(SH_SRC_PARSE_DIR) -I$(SH_SRC_GEN_MATH_DIR) $(SRW_SRC_DEF) 
LDFLAGS=-L$(LIB_DIR) -lm 

ifeq ($(MODE),cuda)
CFLAGS += -D_OFFLOAD_GPU -DUSE_CUDA -D_FFTW3 -I$(NVCOMPILERS)/$(NVARCH)/$(NVVERSION)/math_libs/include -I$(NVCOMPILERS)/$(NVARCH)/$(NVVERSION)/cuda/include -std=c++11 -mavx2 -fno-math-errno
NVCC = nvcc
NVCXX = nvc++
LDFLAGS += -L$(NVCOMPILERS)/$(NVARCH)/$(NVVERSION)/cuda/lib64 -lcudart_static -lcudadevrt -lcufft -lrt
NVCFLAGS = -O3 -g -I$(SRW_SRC_GEN_DIR) -I$(SRW_SRC_LIB_DIR) -I$(SH_SRC_PARSE_DIR) -I$(SH_SRC_GEN_MATH_DIR) $(SRW_SRC_DEF) -D_OFFLOAD_GPU -DUSE_CUDA -D_FFTW3 -arch=sm_86 -dlto -rdc=true -Xcompiler '-fPIC'
CUDA_OBJ= gmfftgpu0.o srradmnpgpu.o srradstrgpu.o sroptelmgpu.o #srradintgpu.o
else
ifeq ($(MODE), omp)
CFLAGS+= -D_WITH_OMP -fopenmp -Wno-write-strings 
LDFLAGS+= -lfftw 
else
ifeq ($(MODE), 0)
CFLAGS+= -D_FFTW3 
LDFLAGS+= -lfftw3f -lfftw3 
else
$(error Unknown SRW compilation option)
endif
endif
endif

#PYPATH=/nsls2/projects/bldev/bldev-env
#PYPATH=/usr
#PYFLAGS=-I$(PYPATH)/include/python3.8 -L$(PYPATH)/lib/python3.8
#PYFLAGS=-I$(PYPATH)/include/python3.2m -L$(PYPATH)/lib/python3.2
#PYFLAGS=-I$(PYPATH)/include/python2.7 -L$(PYPATH)/lib/python2.7
#PYFLAGS=-I$(PYPATH)/include/python2.6 -L$(PYPATH)/lib/python2.6

OBJ=	utidev.o auxparse.o gmfft.o gmfit.o gminterp.o gmmeth.o gmtrans.o srclcuti.o srcradint.o srctrjdt.o sremitpr.o srgsnbm.o srgtrjdt.o srisosrc.o srmagcnt.o srmagfld.o srmatsta.o sroptapt.o sroptcnt.o sroptdrf.o sroptel2.o sroptel3.o sroptelm.o sroptfoc.o sroptgrat.o sroptgtr.o sropthck.o sroptcryst.o sroptmat.o sroptpsh.o sroptshp.o sroptsmr.o sroptwgr.o sroptzp.o sroptzps.o srpersto.o srpowden.o srprdint.o srprgind.o srpropme.o srptrjdt.o srradinc.o srradint.o srradmnp.o srradstr.o srremflp.o srsase.o srsend.o srstowig.o srsysuti.o srthckbm.o srthckbm2.o srtrjaux.o srtrjdat.o srtrjdat3d.o all_com.o check.o diagno.o esource.o field.o incoherent.o initrun.o input.o loadbeam.o loadrad.o magfield.o main.o math.o mpi.o output.o partsim.o pushp.o rpos.o scan.o source.o stepz.o string.o tdepend.o timerec.o track.o srerror.o srwlib.o 

PRG=	libsrw.a

lib:	$(OBJ) $(CUDA_OBJ)
	$(NVCC) $(NVCFLAGS) -dlink -o srwl_link.o $(CUDA_OBJ) $(LDFLAGS)
	ar -cvq $(PRG) *.o
	#cp $(PRG) $(LIB_DIR)/
	rm -f *.o *.co

%.o: $(SRW_SRC_LIB_DIR)/%.cpp
	$(CXX) $(CFLAGS) -c $<  

%.o: $(SH_SRC_PARSE_DIR)/%.cpp
	$(CXX) $(CFLAGS) -c $<  

%.o: $(SH_SRC_GEN_MATH_DIR)/%.cpp
	$(CXX) $(CFLAGS) -c $<  

%.o: $(SRW_SRC_GEN_DIR)/%.cpp
	$(CXX) $(CFLAGS) -c $<  

%.o: $(SRW_SRC_GENESIS_DIR)/%.c
	$(CC) $(CFLAGS) -c $< 
	
%.o: $(SRW_SRC_LIB_DIR)/%.cu
	$(NVCC) $(NVCFLAGS) -c $<  

%.o: $(SH_SRC_PARSE_DIR)/%.cu
	$(NVCC) $(NVCFLAGS) -c $<  

%.o: $(SH_SRC_GEN_MATH_DIR)/%.cu
	$(NVCC) $(NVCFLAGS) -c $<  

%.o: $(SRW_SRC_GEN_DIR)/%.cu
	$(NVCC) $(NVCFLAGS) -c $<  

%.o: $(SRW_SRC_GENESIS_DIR)/%.cu
	$(NVCC) $(NVCFLAGS) -c $<  

pylib:
	$(CXX) -shared $(CFLAGS) $(PYFLAGS) -o srwlpy.so ../src/clients/python/srwlpy.cpp libsrw.a $(LDFLAGS)

srwlclient: $(SRW_SRC_DIR)/clients/c/srwlclient.cpp
	$(CXX) $(CFLAGS) -O3 -o srwlclient ../src/clients/c/srwlclient.cpp libsrw.a $(LDFLAGS)

clean:
	rm -f *.o *.co *.so *.a srwlclient

all: lib pylib srwlclient
