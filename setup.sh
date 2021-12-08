NVARCH=`uname -s`_`uname -m`; export NVARCH
NVCOMPILERS=/opt/nvidia/hpc_sdk; export NVCOMPILERS
MANPATH=$MANPATH:$NVCOMPILERS/$NVARCH/20.11/compilers/man; export MANPATH
PATH=$NVCOMPILERS/$NVARCH/20.11/compilers/bin:$PATH; export PATH
LD_LIBRARY_PATH=$NVCOMPILERS/$NVARCH/20.11/math_libs/lib64:$NVCOMPILERS/$NVARCH/20.11/cuda/lib64:$LD_LIBRARY_PATH; export LD_LIBRARY_PATH
