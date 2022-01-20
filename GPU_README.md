## Build Steps on gpu-002:

```bash
git clone https://github.com/himanshugoel2797/SRW_GPU_Latest.git
conda activate gcc9-conda-forge
cd SRW_GPU_Latest
. setup.sh
make
```

## Perlmutter Notes
```bash
module load PrgEnv-gnu
module load cudatoolkit
module load gcc/10.3.0
module load python
conda create -n srw-test python-3.9 pip numpy scipy scikit-learn matplotlib h5py
conda active srw-test
pip install cupy-cuda114
git clone https://github.com/himanshugoel2797/SRW_GPU_Latest.git
# Uncomment PYPATH=/usr and set PYFLAGS for python3.6m in cpp/gcc/Makefile
cd SRW_GPU_Latest
. setup.sh
make all
```

Note: module load gcc/10.3.0 necessary to workaround a bug in gcc-11 and conda. Eventually may not need it.